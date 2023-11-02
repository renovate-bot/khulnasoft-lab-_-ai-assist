from pathlib import Path
from typing import Any, Optional

from ai_gateway.code_suggestions.base import (
    CodeSuggestionsOutput,
    increment_lang_counter,
    resolve_lang_id,
)
from ai_gateway.code_suggestions.processing import LanguageId, Prompt
from ai_gateway.code_suggestions.processing.post.generations import PostProcessor
from ai_gateway.code_suggestions.processing.pre import (
    PromptBuilderPrefixBased,
    TokenStrategyBase,
)
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import ModelAPICallError, ModelAPIError, TextGenBaseModel
from ai_gateway.prompts import PromptTemplate

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
    ):
        self.model = model

        self.prompt: Optional[Prompt] = None
        self.instrumentator = TextGenModelInstrumentator(
            model.metadata.engine, model.metadata.name
        )
        self.prompt_builder = PromptBuilderPrefixBased(
            model.MAX_MODEL_LEN, tokenization_strategy
        )

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

    def with_prompt_prepared(self, prompt: str):
        self.prompt = self.prompt_builder.wrap(prompt)

    async def execute(
        self,
        prefix: str,
        file_name: str,
        editor_lang: Optional[str] = None,
        **kwargs: Any,
    ) -> CodeSuggestionsOutput:
        lang_id = resolve_lang_id(file_name, editor_lang)
        increment_lang_counter(file_name, lang_id, editor_lang)

        prompt = self._get_prompt(prefix, file_name, lang_id)

        with self.instrumentator.watch(prompt) as watch_container:
            try:
                watch_container.register_lang(lang_id, editor_lang)

                if res := await self.model.generate(prompt.prefix, "", **kwargs):
                    watch_container.register_model_output_length(res.text)
                    watch_container.register_model_score(res.score)
                    watch_container.register_safety_attributes(res.safety_attributes)

                    generation = res.text if kwargs.get('prompt_version') == 2 else PostProcessor(prefix).process(res.text)

                    return CodeSuggestionsOutput(
                        text=generation,
                        score=res.score,
                        model=self.model.metadata,
                        lang_id=lang_id,
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
        )
