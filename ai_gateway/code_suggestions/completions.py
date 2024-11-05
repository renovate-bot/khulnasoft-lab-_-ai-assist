from typing import Any, AsyncIterator, Optional, Union

import structlog
from dependency_injector.providers import Factory

from ai_gateway.code_suggestions.base import (
    CodeSuggestionsChunk,
    CodeSuggestionsOutput,
    LanguageId,
    increment_lang_counter,
    resolve_lang_id,
)
from ai_gateway.code_suggestions.processing import (
    ModelEngineCompletions,
    ModelEngineOutput,
    Prompt,
    TokenStrategyBase,
)
from ai_gateway.code_suggestions.processing.post.completions import PostProcessor
from ai_gateway.code_suggestions.processing.pre import PromptBuilderPrefixBased
from ai_gateway.code_suggestions.processing.typing import MetadataExtraInfo
from ai_gateway.instrumentators import (
    KnownMetrics,
    TextGenModelInstrumentator,
    benchmark,
)
from ai_gateway.models import ChatModelBase, Message, ModelAPICallError, ModelAPIError
from ai_gateway.models.agent_model import AgentModel
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import SnowplowEvent, SnowplowEventContext

__all__ = ["CodeCompletionsLegacy", "CodeCompletions"]

log = structlog.stdlib.get_logger("codesuggestions")


class CodeCompletionsLegacy:
    def __init__(
        self,
        engine: ModelEngineCompletions,
        post_processor: Factory[PostProcessor],
        snowplow_instrumentator: SnowplowInstrumentator,
    ):
        self.engine = engine
        self.post_processor = post_processor
        self.instrumentator = snowplow_instrumentator

    async def execute(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        snowplow_event_context: Optional[SnowplowEventContext] = None,
        **kwargs: Any,
    ) -> list[ModelEngineOutput]:
        responses = await self.engine.generate(
            prefix, suffix, file_name, editor_lang, **kwargs
        )

        self.instrumentator.watch(
            SnowplowEvent(
                context=snowplow_event_context,
                action="tokens_per_user_request_prompt",
                label="code_completion",
                value=responses[0].tokens_consumption_metadata.input_tokens,
            )
        )

        outputs = []
        # Since all metadata objects are the same, take the first one
        tokens_consumption_metadata = responses[0].tokens_consumption_metadata
        total_output_tokens = tokens_consumption_metadata.output_tokens
        for response in responses:
            if not response.text:
                outputs.append(response)
                break

            with benchmark(
                metric_key=KnownMetrics.POST_PROCESSING_DURATION,
                labels={
                    "model_engine": self.engine.model.metadata.engine,
                    "model_name": self.engine.model.metadata.name,
                },
            ):
                processed_completion = await self.post_processor(
                    prefix, suffix=suffix, lang_id=response.lang_id
                ).process(response.text)

            outputs.append(
                ModelEngineOutput(
                    text=processed_completion,
                    score=response.score,
                    model=response.model,
                    lang_id=response.lang_id,
                    metadata=response.metadata,
                    tokens_consumption_metadata=response.tokens_consumption_metadata,
                )
            )

        self.instrumentator.watch(
            SnowplowEvent(
                context=snowplow_event_context,
                action="tokens_per_user_request_response",
                label="code_completion",
                value=total_output_tokens,
            )
        )
        return outputs


class CodeCompletions:
    SUFFIX_RESERVED_PERCENT = 0.07

    def __init__(
        self,
        model: TextGenModelBase,
        tokenization_strategy: TokenStrategyBase,
        post_processor: Optional[Factory[PostProcessor]] = None,
    ):
        self.model = model

        self.instrumentator = TextGenModelInstrumentator(
            model.metadata.engine, model.metadata.name
        )

        self.post_processor = post_processor

        # If you need the previous logic for building prompts using tree-sitter, refer to CodeCompletionsLegacy.
        # In the future, we plan to completely drop CodeCompletionsLegacy and move its logic to CodeCompletions
        # Ref: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/296
        self.prompt_builder = PromptBuilderPrefixBased(
            model.MAX_MODEL_LEN, tokenization_strategy
        )

    def _get_prompt(
        self,
        prefix: str,
        suffix: str,
        raw_prompt: Optional[str | list[Message]] = None,
        code_context: Optional[list] = None,
        context_max_percent: Optional[float] = None,
    ) -> Prompt:
        if raw_prompt:
            return self.prompt_builder.wrap(raw_prompt)

        self.prompt_builder.add_content(
            prefix,
            suffix=suffix,
            suffix_reserved_percent=self.SUFFIX_RESERVED_PERCENT,
            context_max_percent=context_max_percent,
            code_context=code_context,
        )

        prompt = self.prompt_builder.build()

        return prompt

    async def execute(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: Optional[str] = None,
        raw_prompt: Optional[str | list[Message]] = None,
        code_context: Optional[list] = None,
        stream: bool = False,
        snowplow_event_context: Optional[SnowplowEventContext] = None,
        **kwargs: Any,
    ) -> Union[CodeSuggestionsOutput, AsyncIterator[CodeSuggestionsChunk]]:
        lang_id = resolve_lang_id(file_name, editor_lang)
        increment_lang_counter(file_name, lang_id, editor_lang)

        context_max_percent = kwargs.pop(
            "context_max_percent", 1.0
        )  # default is full context window
        prompt = self._get_prompt(
            prefix,
            suffix,
            raw_prompt=raw_prompt,
            code_context=code_context,
            context_max_percent=context_max_percent,
        )

        with self.instrumentator.watch(prompt) as watch_container:
            try:
                watch_container.register_lang(lang_id, editor_lang)

                if isinstance(self.model, AgentModel):
                    params = {"prefix": prompt.prefix, "suffix": prompt.suffix}

                    res = await self.model.generate(params, stream)
                elif isinstance(self.model, ChatModelBase):
                    res = await self.model.generate(
                        prompt.prefix, stream=stream, **kwargs
                    )
                else:
                    res = await self.model.generate(
                        prompt.prefix, prompt.suffix, stream, **kwargs
                    )

                if res:
                    if isinstance(res, AsyncIterator):
                        return self._handle_stream(res)

                    return await self._handle_sync(
                        prompt, res, lang_id, watch_container
                    )
            except ModelAPICallError as ex:
                watch_container.register_model_exception(str(ex), ex.code)
                raise
            except ModelAPIError as ex:
                # TODO: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/294
                watch_container.register_model_exception(str(ex), -1)
                raise

        return CodeSuggestionsOutput(
            text="",
            score=0,
            model=self.model.metadata,
            lang_id=lang_id,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
                tokens_consumption_metadata=self._get_tokens_consumption_metadata(
                    prompt
                ),
            ),
        )

    async def _handle_stream(
        self, response: AsyncIterator[TextGenModelChunk]
    ) -> AsyncIterator[CodeSuggestionsChunk]:
        async for chunk in response:
            chunk_content = CodeSuggestionsChunk(text=chunk.text)
            yield chunk_content

    async def _handle_sync(
        self,
        prompt: Prompt,
        response: TextGenModelOutput,
        lang_id: Optional[LanguageId],
        watch_container: TextGenModelInstrumentator.WatchContainer,
    ) -> CodeSuggestionsOutput:
        watch_container.register_model_output_length(response.text)
        watch_container.register_model_score(response.score)
        watch_container.register_safety_attributes(response.safety_attributes)

        response_text = await self._get_response_text(response.text, prompt, lang_id)

        return CodeSuggestionsOutput(
            text=response_text,
            score=response.score,
            model=self.model.metadata,
            lang_id=lang_id,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
                tokens_consumption_metadata=self._get_tokens_consumption_metadata(
                    prompt, response
                ),
            ),
        )

    async def _get_response_text(
        self, response_text: str, prompt: Prompt, lang_id: LanguageId
    ):
        if self.post_processor:
            return await self.post_processor(
                prompt.prefix, suffix=prompt.suffix, lang_id=lang_id
            ).process(response_text)

        return response_text

    def _get_tokens_consumption_metadata(
        self, prompt: Prompt, response: Optional[TextGenModelOutput] = None
    ) -> TokensConsumptionMetadata:
        input_tokens = sum(
            component.length_tokens for component in prompt.metadata.components.values()
        )

        if response:
            output_tokens = (
                response.metadata.output_tokens
                if response.metadata and hasattr(response.metadata, "output_tokens")
                else 0
            )
        else:
            output_tokens = 0

        context_tokens_sent = 0
        context_tokens_used = 0

        if prompt.metadata.code_context and isinstance(
            prompt.metadata.code_context, MetadataExtraInfo
        ):
            context_tokens_sent = prompt.metadata.code_context.pre.length_tokens
            context_tokens_used = prompt.metadata.code_context.post.length_tokens

        return TokensConsumptionMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_tokens_sent=context_tokens_sent,
            context_tokens_used=context_tokens_used,
        )
