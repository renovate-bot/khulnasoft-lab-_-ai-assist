from pathlib import Path
from typing import Any, NamedTuple, Optional

from ai_gateway.code_suggestions.processing import LanguageId, Prompt
from ai_gateway.code_suggestions.processing.base import LANGUAGE_COUNTER
from ai_gateway.code_suggestions.processing.ops import (
    lang_from_editor_lang,
    lang_from_filename,
)
from ai_gateway.code_suggestions.processing.post.generations import PostProcessor
from ai_gateway.code_suggestions.processing.pre import (
    PromptBuilderPrefixBased,
    TokenStrategyBase,
)
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import (
    ModelMetadata,
    TextGenBaseModel,
    VertexModelInternalError,
    VertexModelInvalidArgument,
)
from ai_gateway.prompts import PromptTemplate

__all__ = ["CodeGenerationsOutput", "CodeGenerations"]


TPL_GENERATION_BASE = """
```{lang}
{prefix}
```
""".strip(
    "\n"
)


class CodeGenerationsOutput(NamedTuple):
    text: str
    score: float
    model: ModelMetadata
    lang_id: Optional[LanguageId] = None

    @property
    def lang(self) -> str:
        return self.lang_id.name.lower() if self.lang_id else ""


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
    ) -> CodeGenerationsOutput:
        lang_id = _resolve_lang_id(file_name, editor_lang)
        _increment_lang_counter(file_name, lang_id, editor_lang)

        prompt = self._get_prompt(prefix, file_name, lang_id)

        with self.instrumentator.watch(prompt) as watch_container:
            try:
                watch_container.register_lang(lang_id, editor_lang)

                if res := await self.model.generate(prompt.prefix, "", **kwargs):
                    watch_container.register_model_output_length(res.text)
                    watch_container.register_model_score(res.score)
                    watch_container.register_safety_attributes(res.safety_attributes)

                    generation = PostProcessor(prefix).process(res.text)

                    return CodeGenerationsOutput(
                        text=generation,
                        score=res.score,
                        model=self.model.metadata,
                        lang_id=lang_id,
                    )

            except (VertexModelInvalidArgument, VertexModelInternalError) as ex:
                watch_container.register_model_exception(str(ex), ex.code)

        return CodeGenerationsOutput(
            text="",
            score=0,
            model=self.model.metadata,
            lang_id=lang_id,
        )


def _resolve_lang_id(file_name: str, editor_lang: str) -> Optional[LanguageId]:
    lang_id = lang_from_filename(file_name)

    if lang_id is None and editor_lang:
        lang_id = lang_from_editor_lang(editor_lang)

    return lang_id


# TODO: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/292
def _increment_lang_counter(
    filename: str,
    lang_id: Optional[LanguageId] = None,
    editor_lang_id: Optional[str] = None,
):
    labels = {"lang": None, "editor_lang": None}

    if lang_id:
        labels["lang"] = lang_id.name.lower()

    if editor_lang_id:
        labels["editor_lang"] = editor_lang_id

    labels["extension"] = Path(filename).suffix[1:]

    LANGUAGE_COUNTER.labels(**labels).inc()
