from typing import Any, AsyncIterator, Optional, Union

from ai_gateway.code_suggestions import CodeSuggestionsChunk, CodeSuggestionsOutput
from ai_gateway.code_suggestions.base import increment_lang_counter, resolve_lang_id
from ai_gateway.code_suggestions.processing import (
    ModelEngineCompletions,
    ModelEngineOutput,
    Prompt,
)
from ai_gateway.code_suggestions.processing.pre import (
    PromptBuilderPrefixBased,
    TokenStrategyBase,
)
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import ModelAPICallError, ModelAPIError, TextGenBaseModel

__all__ = ["CodeCompletionsLegacy", "CodeCompletions"]


class CodeCompletionsLegacy:
    def __init__(self, engine: ModelEngineCompletions):
        self.engine = engine

    async def execute(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        language_identifier: str,
        **_kwargs: Any,
    ) -> ModelEngineOutput:
        suggestion = await self.engine.generate(
            prefix, suffix, file_name, language_identifier
        )

        return suggestion


class CodeCompletions:
    SUFFIX_RESERVED_PERCENT = 0.07

    def __init__(
        self, model: TextGenBaseModel, tokenization_strategy: TokenStrategyBase
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
                    if stream:
                        return self._handle_stream(res)

                    watch_container.register_model_output_length(res.text)
                    watch_container.register_model_score(res.score)
                    watch_container.register_safety_attributes(res.safety_attributes)

                    return CodeSuggestionsOutput(
                        text=res.text,
                        score=res.score,
                        model=self.model.metadata,
                        lang_id=lang_id,
                        metadata=CodeSuggestionsOutput.Metadata(
                            experiments=[],
                        ),
                    )

            except ModelAPICallError as ex:
                watch_container.register_model_exception(str(ex), ex.code)
            except ModelAPIError as ex:
                # TODO: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/294
                watch_container.register_model_exception(str(ex), -1)

        return CodeSuggestionsOutput(
            text="",
            score=0,
            model=self.model.metadata,
            lang_id=lang_id,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
            ),
        )

    async def _handle_stream(self, response) -> AsyncIterator[CodeSuggestionsChunk]:
        async for chunk in response:
            chunk_content = CodeSuggestionsChunk(text=chunk.text)
            yield chunk_content
