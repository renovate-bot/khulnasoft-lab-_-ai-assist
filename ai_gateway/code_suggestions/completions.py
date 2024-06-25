from typing import Any, AsyncIterator, Optional, Union

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
from ai_gateway.instrumentators import (
    KnownMetrics,
    TextGenModelInstrumentator,
    benchmark,
)
from ai_gateway.models import ModelAPICallError, ModelAPIError
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import SnowplowEvent

__all__ = ["CodeCompletionsLegacy", "CodeCompletions"]


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
        **kwargs: Any,
    ) -> ModelEngineOutput:
        responses = await self.engine.generate(
            prefix, suffix, file_name, editor_lang, **kwargs
        )

        self.instrumentator.watch(
            SnowplowEvent(
                context=None,
                action="tokens_per_user_request_prompt",
                label="code_completion",
                value=responses[0].tokens_consumption_metadata.input_tokens,
            )
        )

        outputs = []
        output_tokens = 0
        for response in responses:
            output_tokens += response.tokens_consumption_metadata.output_tokens
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
                context=None,
                action="tokens_per_user_request_response",
                label="code_completion",
                value=output_tokens,
            )
        )
        return outputs


class CodeCompletions:
    SUFFIX_RESERVED_PERCENT = 0.07

    def __init__(
        self, model: TextGenModelBase, tokenization_strategy: TokenStrategyBase
    ):
        self.model = model

        self.instrumentator = TextGenModelInstrumentator(
            model.metadata.engine, model.metadata.name
        )

        # If you need the previous logic for building prompts using tree-sitter, refer to CodeCompletionsLegacy.
        # In the future, we plan to completely drop CodeCompletionsLegacy and move its logic to CodeCompletions
        # Ref: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/296
        self.prompt_builder = PromptBuilderPrefixBased(
            model.MAX_MODEL_LEN, tokenization_strategy
        )

    def _get_prompt(
        self, prefix: str, suffix: str, raw_prompt: Optional[str] = None
    ) -> Prompt:
        if raw_prompt:
            return self.prompt_builder.wrap(raw_prompt)

        self.prompt_builder.add_content(
            prefix, suffix=suffix, suffix_reserved_percent=self.SUFFIX_RESERVED_PERCENT
        )
        prompt = self.prompt_builder.build()

        return prompt

    async def execute(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: Optional[str] = None,
        raw_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[CodeSuggestionsOutput, AsyncIterator[CodeSuggestionsChunk]]:
        lang_id = resolve_lang_id(file_name, editor_lang)
        increment_lang_counter(file_name, lang_id, editor_lang)

        prompt = self._get_prompt(prefix, suffix, raw_prompt=raw_prompt)

        with self.instrumentator.watch(prompt) as watch_container:
            try:
                watch_container.register_lang(lang_id, editor_lang)

                if res := await self.model.generate(
                    prompt.prefix, prompt.suffix, stream, **kwargs
                ):
                    if isinstance(res, AsyncIterator):
                        return self._handle_stream(res)

                    return self._handle_sync(res, lang_id, watch_container)
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
            ),
        )

    async def _handle_stream(
        self, response: AsyncIterator[TextGenModelChunk]
    ) -> AsyncIterator[CodeSuggestionsChunk]:
        async for chunk in response:
            chunk_content = CodeSuggestionsChunk(text=chunk.text)
            yield chunk_content

    def _handle_sync(
        self,
        response: TextGenModelOutput,
        lang_id: Optional[LanguageId],
        watch_container: TextGenModelInstrumentator.WatchContainer,
    ) -> CodeSuggestionsOutput:
        watch_container.register_model_output_length(response.text)
        watch_container.register_model_score(response.score)
        watch_container.register_safety_attributes(response.safety_attributes)

        return CodeSuggestionsOutput(
            text=response.text,
            score=response.score,
            model=self.model.metadata,
            lang_id=lang_id,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
            ),
        )
