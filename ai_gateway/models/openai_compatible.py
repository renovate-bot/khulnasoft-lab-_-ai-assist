from enum import Enum
from typing import Any, AsyncIterator, Callable, Union

import httpx
import structlog
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAPIResponse,
    AsyncOpenAI,
)
from openai._types import NOT_GIVEN

from ai_gateway.models.base import (
    KindModelProvider,
    ModelAPICallError,
    ModelAPIError,
    ModelMetadata,
    SafetyAttributes,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.models.base_chat import ChatModelBase, Message

__all__ = [
    "OpenAiAPIConnectionError",
    "OpenAiAPIStatusError",
    "OpenAiAPITimeoutError",
    "OpenAiCompatibleModel",
    "KindOpenAiCompatibleModel",
]

log = structlog.stdlib.get_logger("codesuggestions")


class OpenAiAPIConnectionError(ModelAPIError):
    @classmethod
    def from_exception(cls, ex: APIConnectionError):
        wrapper = cls(ex.message, errors=(ex,))

        return wrapper


class OpenAiAPIStatusError(ModelAPICallError):
    @classmethod
    def from_exception(cls, ex: APIStatusError):
        cls.code = ex.status_code
        wrapper = cls(ex.message, errors=(ex,))

        return wrapper


class OpenAiAPITimeoutError(ModelAPIError):
    @classmethod
    def from_exception(cls, ex: APITimeoutError):
        wrapper = cls(ex.message, errors=(ex,))

        return wrapper


class KindOpenAiCompatibleModel(str, Enum):
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"


class OpenAiCompatibleModel(ChatModelBase):
    MAX_MODEL_LEN = 32_000

    OPTS_CLIENT = {
        "default_headers": {},
        "max_retries": 1,
        "base_url": "http://localhost:11434/v1",
    }

    OPTS_MODEL = {
        "timeout": httpx.Timeout(30.0, connect=5.0),
        "max_tokens": 2048,
        "temperature": 0.2,
        "stop": NOT_GIVEN,
    }

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str = KindOpenAiCompatibleModel.MISTRAL.value,
        **kwargs: Any,
    ):
        client_opts = _obtain_opts(OpenAiCompatibleModel.OPTS_CLIENT, **kwargs)
        self.client = client.with_options(**client_opts)

        self.model_opts = _obtain_opts(OpenAiCompatibleModel.OPTS_MODEL, **kwargs)
        self._metadata = ModelMetadata(
            name=model_name,
            engine=KindModelProvider.OPENAI_COMPATIBLE.value,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(
        self,
        messages: list[Message],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[TextGenModelOutput, AsyncIterator[TextGenModelChunk]]:
        opts = _obtain_opts(self.model_opts, **kwargs)
        log.debug("codegen openai call:", **opts)

        messages = [message.model_dump(mode="json") for message in messages]
        with self.instrumentator.watch(stream=stream) as watcher:
            try:
                suggestion = await self.client.chat.completions.create(
                    model=self.metadata.name,
                    messages=messages,
                    stream=stream,
                    **opts,
                )
            except APIStatusError as ex:
                raise OpenAiAPIStatusError.from_exception(ex)
            except APITimeoutError as ex:
                raise OpenAiAPITimeoutError.from_exception(ex)
            except APIConnectionError as ex:
                raise OpenAiAPIConnectionError.from_exception(ex)

            if stream:
                return self._handle_stream(suggestion, lambda: watcher.finish())

        return TextGenModelOutput(
            text=suggestion.choices[0].message.content,
            # Give a high value, the model doesn't return scores.
            score=10**5,
            safety_attributes=SafetyAttributes(),
        )

    async def _handle_stream(
        self, response: AsyncAPIResponse, after_callback: Callable
    ) -> AsyncIterator[TextGenModelChunk]:
        try:
            async for chunk in response:
                yield TextGenModelChunk(text=(chunk.choices[0].delta.content or ""))
        finally:
            after_callback()

    @classmethod
    def from_model_name(
        cls,
        name: Union[str, KindOpenAiCompatibleModel],
        client: AsyncOpenAI,
        **kwargs: Any,
    ):
        try:
            kind_model = KindOpenAiCompatibleModel(name)
        except ValueError:
            raise ValueError(f"no model found by the name '{name}'")

        return cls(client, model_name=kind_model.value, **kwargs)


def _obtain_opts(default_opts: dict, **kwargs: Any) -> dict:
    return {
        opt_name: kwargs.pop(opt_name, opt_value) or opt_value
        for opt_name, opt_value in default_opts.items()
    }
