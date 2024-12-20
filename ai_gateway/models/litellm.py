from enum import StrEnum
from typing import AsyncIterator, Callable, Optional, Sequence, Union

from litellm import CustomStreamWrapper, ModelResponse, acompletion
from litellm.exceptions import APIConnectionError, InternalServerError
from openai import AsyncOpenAI

from ai_gateway.models.base import (
    KindModelProvider,
    ModelAPIError,
    ModelMetadata,
    TokensConsumptionMetadata,
)
from ai_gateway.models.base_chat import ChatModelBase, Message, Role
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.safety_attributes import SafetyAttributes
from ai_gateway.tracking import SnowplowEventContext

__all__ = [
    "LiteLlmChatModel",
    "LiteLlmTextGenModel",
    "KindLiteLlmModel",
    "LiteLlmAPIConnectionError",
    "LiteLlmInternalServerError",
]

STUBBED_API_KEY = "stubbed-api-key"


class LiteLlmAPIConnectionError(ModelAPIError):
    @classmethod
    def from_exception(cls, ex: APIConnectionError):
        wrapper = cls(ex.message, errors=(ex,))

        return wrapper


class LiteLlmInternalServerError(ModelAPIError):
    @classmethod
    def from_exception(cls, ex: InternalServerError):
        wrapper = cls(ex.message, errors=(ex,))

        return wrapper


class KindLiteLlmModel(StrEnum):
    CODEGEMMA = "codegemma"
    CODELLAMA = "codellama"
    CODESTRAL = "codestral"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    DEEPSEEKCODER = "deepseekcoder"
    CLAUDE_3 = "claude_3"
    CLAUDE_3_5 = "claude_3.5"
    GPT = "gpt"
    QWEN_2_5 = "qwen2p5-coder-7b"

    def _chat_provider_prefix(self, provider):
        # Chat models hosted behind openai proxies should be prefixed with "openai/":
        # https://docs.litellm.ai/docs/providers/openai_compatible
        if provider == KindModelProvider.LITELLM:
            return "custom_openai"

        return provider.value

    def _text_provider_prefix(self, provider):
        # Text completion models hosted behind openai proxies should be prefixed with "text-completion-openai/":
        # https://docs.litellm.ai/docs/providers/openai_compatible
        if provider == KindModelProvider.LITELLM:
            return "text-completion-custom_openai"

        return f"text-completion-{provider.value}"

    def chat_model(self, provider=KindModelProvider.LITELLM) -> str:
        return f"{self._chat_provider_prefix(provider)}/{self.value}"

    def text_model(self, provider=KindModelProvider.LITELLM) -> str:
        return f"{self._text_provider_prefix(provider)}/{self.value}"


class ModelCompletionType(StrEnum):
    TEXT = "text"
    CHAT = "chat"
    FIM = "fim"


MODEL_STOP_TOKENS = {
    KindLiteLlmModel.MISTRAL: ["</new_code>"],
    # Ref: https://huggingface.co/google/codegemma_2b-7b
    # The model returns the completion, followed by one of the FIM tokens or the EOS token.
    # You should ignore everything that comes after any of these tokens.
    KindLiteLlmModel.CODEGEMMA: [
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|file_separator|>",
    ],
    KindLiteLlmModel.QWEN_2_5: [
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<|im_start|>",
        "<|im_end|>",
        "\n\n",
    ],
}

MODEL_SPECIFICATIONS = {
    KindLiteLlmModel.QWEN_2_5: {
        "timeout": 60,
        "completion_type": ModelCompletionType.FIM,
        "fim_tokens": {
            "prefix": "<|fim_prefix|>",
            "suffix": "<|fim_suffix|>",
            "middle": "<|fim_middle|>",
        },
        "session_header": True,
    },
}


class LiteLlmChatModel(ChatModelBase):
    @property
    def MAX_MODEL_LEN(self):  # pylint: disable=invalid-name
        codegemma_models = {
            KindLiteLlmModel.CODEGEMMA,
        }

        codelama_models = {
            KindLiteLlmModel.CODELLAMA,
        }

        if self.model_name in codegemma_models:
            return 8_192

        if self.model_name in codelama_models:
            return 16_384

        return 32_768

    def __init__(
        self,
        model_name: KindLiteLlmModel = KindLiteLlmModel.MISTRAL,
        provider: Optional[KindModelProvider] = KindModelProvider.LITELLM,
        metadata: Optional[ModelMetadata] = None,
        disable_streaming: bool = False,
        async_fireworks_client: Optional[AsyncOpenAI] = None,
    ):
        self._metadata = _init_litellm_model_metadata(metadata, model_name, provider)
        self.provider = provider
        self.model_name = model_name
        self.stop_tokens = MODEL_STOP_TOKENS.get(model_name, [])
        self.disable_streaming = disable_streaming
        self.async_fireworks_client = async_fireworks_client

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(
        self,
        messages: list[Message],
        stream: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        top_p: float = 0.95,
        code_context: Optional[Sequence[str]] = None,
    ) -> Union[TextGenModelOutput, AsyncIterator[TextGenModelChunk]]:
        should_stream = not self.disable_streaming and stream

        if isinstance(messages, str):
            messages = [Message(content=messages, role=Role.USER)]

        litellm_messages = [message.model_dump(mode="json") for message in messages]

        completion_args = {
            "messages": litellm_messages,
            "stream": should_stream,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens,
            "timeout": 30.0,
            "stop": self.stop_tokens,
            **self.model_metadata_to_params(),
        }

        if self.provider == KindModelProvider.FIREWORKS:
            completion_args["client"] = self.async_fireworks_client
            # disable prompt caching
            completion_args["prompt_cache_max_len"] = 0

        with self.instrumentator.watch(stream=stream) as watcher:
            suggestion = await acompletion(**completion_args)

            if should_stream:
                return self._handle_stream(
                    suggestion,
                    watcher.finish,
                    watcher.register_error,
                )

        return TextGenModelOutput(
            text=suggestion.choices[0].message.content,
            # Give a high value, the model doesn't return scores.
            score=10**5,
            safety_attributes=SafetyAttributes(),
            metadata=self._extract_suggestion_metadata(suggestion),
        )

    async def _handle_stream(
        self,
        response: CustomStreamWrapper,
        after_callback: Callable,
        error_callback: Callable,
    ) -> AsyncIterator[TextGenModelChunk]:
        try:
            async for chunk in response:
                yield TextGenModelChunk(text=(chunk.choices[0].delta.content or ""))
        except Exception:
            error_callback()
            raise
        finally:
            after_callback()

    def _extract_suggestion_metadata(self, suggestion):
        return TokensConsumptionMetadata(
            output_tokens=(
                suggestion.usage.completion_tokens
                if hasattr(suggestion, "usage")
                else 0
            ),
        )

    @classmethod
    def from_model_name(
        cls,
        name: Union[str, KindLiteLlmModel],
        custom_models_enabled: bool = False,
        disable_streaming: bool = False,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        identifier: Optional[str] = None,
        provider: Optional[KindModelProvider] = KindModelProvider.LITELLM,
        provider_keys: Optional[dict] = None,
        provider_endpoints: Optional[dict] = None,
        async_fireworks_client: Optional[AsyncOpenAI] = None,
    ):
        if not custom_models_enabled and provider == KindModelProvider.LITELLM:
            if endpoint is not None or api_key is not None:
                raise ValueError("specifying custom models endpoint is disabled")

        if provider == KindModelProvider.MISTRALAI:
            api_key = provider_keys.get("mistral_api_key")

        if provider == KindModelProvider.FIREWORKS:
            api_key = provider_keys.get("fireworks_api_key")

            endpoint, identifier = _get_fireworks_config(provider_endpoints)
            identifier = f"fireworks_ai/{identifier}"

        try:
            kind_model = KindLiteLlmModel(name)
        except ValueError:
            raise ValueError(f"no model found by the name '{name}'")

        model_metadata = ModelMetadata(
            name=kind_model.chat_model(provider),
            engine=provider,
            endpoint=endpoint,
            api_key=api_key,
            identifier=identifier,
        )

        return cls(
            kind_model,
            provider,
            model_metadata,
            disable_streaming,
            async_fireworks_client=async_fireworks_client,
        )


class LiteLlmTextGenModel(TextGenModelBase):
    @property
    def MAX_MODEL_LEN(self):  # pylint: disable=invalid-name
        if self.model_name == KindLiteLlmModel.CODEGEMMA:
            return 8_192

        if self.model_name == KindLiteLlmModel.CODELLAMA:
            return 16_384

        return 32_768

    def __init__(
        self,
        model_name: KindLiteLlmModel = KindLiteLlmModel.CODEGEMMA,
        provider: Optional[KindModelProvider] = KindModelProvider.LITELLM,
        metadata: Optional[ModelMetadata] = None,
        disable_streaming: bool = False,
        async_fireworks_client: Optional[AsyncOpenAI] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self._metadata = _init_litellm_model_metadata(metadata, model_name, provider)
        self.disable_streaming = disable_streaming

        self.stop_tokens = MODEL_STOP_TOKENS.get(model_name, [])
        self.async_fireworks_client = async_fireworks_client

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def specifications(self):
        return MODEL_SPECIFICATIONS.get(self.model_name, {})

    async def generate(
        self,
        prefix: str,
        suffix: Optional[str] = "",
        stream: bool = False,
        temperature: float = 0.95,
        max_output_tokens: int = 16,
        top_p: float = 0.95,
        code_context: Optional[Sequence[str]] = None,
        snowplow_event_context: Optional[SnowplowEventContext] = None,
    ) -> Union[TextGenModelOutput, AsyncIterator[TextGenModelChunk]]:
        should_stream = not self.disable_streaming and stream

        with self.instrumentator.watch(stream=should_stream) as watcher:
            try:
                suggestion = await self._get_suggestion(
                    prefix=prefix,
                    suffix=suffix,
                    stream=should_stream,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_p=top_p,
                    snowplow_event_context=snowplow_event_context,
                )
            except APIConnectionError as ex:
                raise LiteLlmAPIConnectionError.from_exception(ex)
            except InternalServerError as ex:
                raise LiteLlmInternalServerError.from_exception(ex)

            if should_stream:
                return self._handle_stream(
                    suggestion,
                    watcher.finish,
                    watcher.register_error,
                )

        return TextGenModelOutput(
            text=self._extract_suggestion_text(suggestion),
            # Give a high value, the model doesn't return scores.
            score=10**5,
            safety_attributes=SafetyAttributes(),
            metadata=self._extract_suggestion_metadata(suggestion),
        )

    async def _handle_stream(
        self,
        response: CustomStreamWrapper,
        after_callback: Callable,
        error_callback: Callable,
    ) -> AsyncIterator[TextGenModelChunk]:
        try:
            async for chunk in response:
                yield TextGenModelChunk(text=(chunk.choices[0].delta.content or ""))
        except Exception:
            error_callback()
            raise
        finally:
            after_callback()

    async def _get_suggestion(
        self,
        prefix: str,
        stream: bool,
        temperature: float,
        max_output_tokens: int,
        top_p: float,
        suffix: Optional[str] = "",
        snowplow_event_context: Optional[SnowplowEventContext] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        content = prefix

        if self._completion_type() == ModelCompletionType.FIM:
            fim_tokens = MODEL_SPECIFICATIONS.get(self.model_name).get("fim_tokens")
            content = (
                fim_tokens.get("prefix", "")
                + prefix
                + fim_tokens.get("suffix", "")
                + (suffix or "")
                + fim_tokens.get("middle", "")
            )

        completion_args = {
            "messages": [{"content": content, "role": Role.USER}],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "timeout": self.specifications.get("timeout", 30.0),
            "stop": self._get_stop_tokens(),
        }

        completion_args = completion_args | self.model_metadata_to_params()

        if self._completion_type() == ModelCompletionType.TEXT:
            completion_args["suffix"] = suffix
            completion_args["text_completion"] = True

        if (
            self._session_header()
            and snowplow_event_context
            and snowplow_event_context.gitlab_global_user_id
        ):
            completion_args["extra_headers"] = {
                "x-session-affinity": snowplow_event_context.gitlab_global_user_id
            }

        if self.provider == KindModelProvider.FIREWORKS:
            completion_args["client"] = self.async_fireworks_client
            # disable prompt caching
            completion_args["prompt_cache_max_len"] = 0

        return await acompletion(**completion_args)

    def _completion_type(self):
        return self.specifications.get("completion_type", ModelCompletionType.CHAT)

    def _session_header(self):
        return self.specifications.get("session_header", False)

    def _extract_suggestion_text(self, suggestion):
        if self._completion_type() == ModelCompletionType.TEXT:
            return suggestion.choices[0].text

        return suggestion.choices[0].message.content

    def _extract_suggestion_metadata(self, suggestion):
        return TokensConsumptionMetadata(
            output_tokens=(
                suggestion.usage.completion_tokens
                if hasattr(suggestion, "usage")
                else 0
            ),
        )

    def _get_stop_tokens(self):
        return self.stop_tokens

    @classmethod
    def from_model_name(
        cls,
        name: Union[str, KindLiteLlmModel],
        custom_models_enabled: bool = False,
        disable_streaming: bool = False,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        identifier: Optional[str] = None,
        provider: Optional[KindModelProvider] = KindModelProvider.LITELLM,
        provider_keys: Optional[dict] = None,
        provider_endpoints: Optional[dict] = None,
        async_fireworks_client: Optional[AsyncOpenAI] = None,
    ):
        if endpoint is not None or api_key is not None:
            if not custom_models_enabled and provider == KindModelProvider.LITELLM:
                raise ValueError("specifying custom models endpoint is disabled")

        if provider == KindModelProvider.MISTRALAI:
            api_key = provider_keys.get("mistral_api_key")

        if provider == KindModelProvider.FIREWORKS:
            api_key = provider_keys.get("fireworks_api_key")

            if not api_key:
                raise ValueError("Fireworks API key is missing from configuration.")

            endpoint, identifier = _get_fireworks_config(provider_endpoints)
            identifier = f"text-completion-openai/{identifier}"

        try:
            kind_model = KindLiteLlmModel(name)
        except ValueError:
            raise ValueError(f"no model found by the name '{name}'")

        metadata = ModelMetadata(
            name=kind_model.text_model(provider),
            engine=provider.value,
            endpoint=endpoint,
            api_key=api_key,
            identifier=identifier,
        )

        return cls(
            model_name=kind_model,
            provider=provider,
            metadata=metadata,
            disable_streaming=disable_streaming,
            async_fireworks_client=async_fireworks_client,
        )


def _get_fireworks_config(provider_endpoints: dict) -> tuple[str, str]:
    """Get Fireworks endpoint and identifier based on region configuration.

    Args:
        provider_endpoints: Dictionary containing provider endpoint configurations

    Returns:
        tuple: (endpoint, identifier) for Fireworks configuration

    Raises:
        ValueError: If required configuration is missing
    """
    # Get endpoint configuration for selected region
    region_config = provider_endpoints.get("fireworks_current_region_endpoint", {})

    if not region_config:
        raise ValueError("Fireworks regional endpoints configuration is missing.")

    endpoint = region_config.get("endpoint")
    identifier = region_config.get("identifier")

    if not endpoint or not identifier:
        raise ValueError("Fireworks endpoint or identifier missing in region config.")

    return endpoint, identifier


def _init_litellm_model_metadata(
    metadata: Optional[ModelMetadata] = None,
    model_name: KindLiteLlmModel = KindLiteLlmModel.MISTRAL,
    provider: Optional[KindModelProvider] = KindModelProvider.LITELLM,
) -> ModelMetadata:
    if metadata:
        return ModelMetadata(
            **(metadata._asdict() | {"api_key": metadata.api_key or STUBBED_API_KEY})
        )

    return ModelMetadata(
        name=model_name.chat_model(provider),
        api_key=STUBBED_API_KEY,
        engine=provider.value(),
    )
