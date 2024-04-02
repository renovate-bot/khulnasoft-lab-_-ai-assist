from pathlib import Path
from typing import Any, AsyncIterator, Optional, Union

from ai_gateway.code_suggestions.base import (
    CodeSuggestionsChunk,
    CodeSuggestionsOutput,
    ModelProvider,
    increment_lang_counter,
    resolve_lang_id,
)
from ai_gateway.code_suggestions.processing import LanguageId, Prompt, TokenStrategyBase
from ai_gateway.code_suggestions.processing.post.generations import (
    PostProcessor,
    PostProcessorAnthropic,
)
from ai_gateway.code_suggestions.processing.pre import PromptBuilderPrefixBased
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import (
    Message,
    ModelAPICallError,
    ModelAPIError,
    TextGenBaseModel,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.models.chat_model_base import ChatModelBase
from ai_gateway.prompts import PromptTemplate
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import SnowplowEvent

__all__ = ["CodeGenerations"]


TPL_GENERATION_BASE = """
```{lang}
{prefix}
```
""".strip(
    "\n"
)


class CodeGenerations:
    def __init__(
        self,
        model: TextGenBaseModel,
        tokenization_strategy: TokenStrategyBase,
        snowplow_instrumentator: SnowplowInstrumentator,
    ):
        self.model = model

        self.prompt: Optional[Prompt] = None
        self.instrumentator = TextGenModelInstrumentator(
            model.metadata.engine, model.metadata.name
        )
        self.prompt_builder = PromptBuilderPrefixBased(
            model.MAX_MODEL_LEN, tokenization_strategy
        )
        self.tokenization_strategy = tokenization_strategy
        self.snowplow_instrumentator = snowplow_instrumentator

    def _get_prompt(
        self, prefix: str, file_name: str, lang_id: Optional[LanguageId] = None
    ) -> Prompt:
        if self.prompt:
            return self.prompt

        # We use either the language name or the file extension
        # if we couldn't determine the language before
        lang_repl = (
            lang_id.name.lower() if lang_id else Path(file_name).suffix.replace(".", "")
        )

        self.prompt_builder.add_template(
            PromptTemplate(TPL_GENERATION_BASE),
            lang=lang_repl,
        )
        self.prompt_builder.add_content(prefix)
        prompt = self.prompt_builder.build()

        return prompt

    def with_prompt_prepared(self, prompt: str | list[Message]):
        self.prompt = self.prompt_builder.wrap(prompt)

    async def execute(
        self,
        prefix: str,
        file_name: str,
        editor_lang: Optional[str] = None,
        model_provider: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[CodeSuggestionsOutput, AsyncIterator[CodeSuggestionsChunk]]:
        lang_id = resolve_lang_id(file_name, editor_lang)
        increment_lang_counter(file_name, lang_id, editor_lang)

        prompt = self._get_prompt(prefix, file_name, lang_id)

        self.snowplow_instrumentator.watch(
            SnowplowEvent(
                context=None,
                action="tokens_per_user_request_prompt",
                label="code_generation",
                value=sum(
                    md.length_tokens for md in prompt.metadata.components.values()
                ),
            )
        )

        with self.instrumentator.watch(prompt) as watch_container:
            try:
                watch_container.register_lang(lang_id, editor_lang)

                if isinstance(self.model, ChatModelBase):
                    res = await self.model.generate(
                        prompt.prefix, stream=stream, **kwargs
                    )
                else:
                    res = await self.model.generate(
                        prompt.prefix, "", stream=stream, **kwargs
                    )

                if res:
                    if isinstance(res, AsyncIterator):
                        return self._handle_stream(res)

                    return await self._handle_sync(
                        response=res,
                        lang_id=lang_id,
                        model_provider=model_provider,
                        prefix=prefix,
                        watch_container=watch_container,
                    )

            except ModelAPICallError as ex:
                watch_container.register_model_exception(str(ex), ex.code)
            except ModelAPIError as ex:
                # TODO: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/294
                watch_container.register_model_exception(str(ex), -1)

        return CodeSuggestionsOutput(
            text="", score=0, model=self.model.metadata, lang_id=lang_id
        )

    async def _handle_stream(
        self, response: AsyncIterator[TextGenModelChunk]
    ) -> AsyncIterator[CodeSuggestionsChunk]:
        chunks = []
        try:
            async for chunk in response:
                chunk_content = CodeSuggestionsChunk(text=chunk.text)
                chunks.append(chunk.text)
                yield chunk_content
        finally:
            self.snowplow_instrumentator.watch(
                SnowplowEvent(
                    context=None,
                    action="tokens_per_user_request_response",
                    label="code_generation",
                    value=sum(self.tokenization_strategy.estimate_length(chunks)),
                )
            )

    async def _handle_sync(
        self,
        response: TextGenModelOutput,
        lang_id: Optional[LanguageId],
        prefix: str,
        watch_container: TextGenModelInstrumentator.WatchContainer,
        model_provider: Optional[str] = None,
    ) -> CodeSuggestionsOutput:
        watch_container.register_model_output_length(response.text)
        watch_container.register_model_score(response.score)
        watch_container.register_safety_attributes(response.safety_attributes)

        processor = (
            PostProcessorAnthropic
            if model_provider == ModelProvider.ANTHROPIC
            else PostProcessor
        )
        generation = await processor(prefix).process(response.text)

        self.snowplow_instrumentator.watch(
            SnowplowEvent(
                context=None,
                action="tokens_per_user_request_response",
                label="code_generation",
                value=self.tokenization_strategy.estimate_length(response.text)[0],
            )
        )

        return CodeSuggestionsOutput(
            text=generation,
            score=response.score,
            model=self.model.metadata,
            lang_id=lang_id,
        )
