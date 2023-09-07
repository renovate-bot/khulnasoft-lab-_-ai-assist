from pathlib import Path
from typing import Any, Optional

from dependency_injector.providers import Factory
from transformers import PreTrainedTokenizer

from codesuggestions.models import (
    PalmCodeGenBaseModel,
    VertexModelInternalError,
    VertexModelInvalidArgument,
)
from codesuggestions.prompts import PromptTemplate
from codesuggestions.suggestions.processing.base import (
    CodeContent,
    MetadataModel,
    MetadataPromptBuilder,
    ModelEngineBase,
    ModelEngineOutput,
    Prompt,
    PromptBuilderBase,
)
from codesuggestions.suggestions.processing.ops import LanguageId, truncate_content

__all__ = [
    "TPL_GENERATION_BASE",
    "PromptBuilder",
    "ModelEngineGenerations",
]

from codesuggestions.suggestions.processing.post.generations import PostProcessor

TPL_GENERATION_BASE = """
```{lang}
{prefix}
```
""".strip(
    "\n"
)


class PromptBuilder(PromptBuilderBase):
    def __init__(
        self, prefix: CodeContent, file_name: str, lang_id: Optional[LanguageId] = None
    ):
        super().__init__(prefix, lang_id=lang_id)

        self.file_name = file_name

    def add_template(self, tpl: PromptTemplate, **kwargs: Any):
        # We use either the language name or the file extension
        # if we couldn't determine the language before
        lang_repl = (
            self.lang_id.name.lower()
            if self.lang_id
            else Path(self.file_name).suffix.replace(".", "")
        )

        # TODO: We're not building a context right now
        # TODO: so let's put everything in the prefix as a short term solution
        self._prefix = tpl.apply(prefix=self._prefix, lang=lang_repl, **kwargs)

    def build(self) -> Prompt:
        return Prompt(
            prefix=self._prefix,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": self._metadata["prefix"],
                }
            ),
        )


class ModelEngineGenerations(ModelEngineBase):
    def __init__(
        self,
        model: PalmCodeGenBaseModel,
        tokenizer: PreTrainedTokenizer,
        post_processor: Factory[PostProcessor],
    ):
        super().__init__(model, tokenizer)
        self.post_processor_factory = post_processor

    async def _generate(
        self,
        prefix: str,
        _suffix: str,
        file_name: str,
        lang_id: Optional[LanguageId] = None,
        prompt_input: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelEngineOutput:
        model_metadata = MetadataModel(
            name=self.model.model_name, engine=self.model.model_engine
        )
        prompt = self._build_prompt(
            prefix, file_name, lang_id=lang_id, prompt_input=prompt_input
        )

        with self.instrumentator.watch(prompt) as watch_container:
            try:
                if res := await self.model.generate(prompt.prefix, "", **kwargs):
                    watch_container.register_model_output_length(res.text)
                    watch_container.register_model_score(res.score)

                    # TODO: Move the call to the use case class
                    generation = self.post_processor_factory().process(res.text)

                    return ModelEngineOutput(
                        text=generation,
                        score=res.score,
                        model=model_metadata,
                        lang_id=lang_id,
                        metadata=prompt.metadata,
                    )

            except (VertexModelInvalidArgument, VertexModelInternalError) as ex:
                watch_container.register_model_exception(str(ex), ex.code)

        return ModelEngineOutput(
            text="",
            score=0,
            model=model_metadata,
            metadata=MetadataPromptBuilder(components={}),
        )

    def _build_prompt(
        self,
        prefix: str,
        file_name: str,
        lang_id: Optional[LanguageId] = None,
        prompt_input: Optional[str] = None,
    ):
        # If a prompt input was provided, we can skip the pre-processing
        if prompt_input:
            return Prompt(
                prefix=prompt_input, metadata=MetadataPromptBuilder(components={})
            )

        tpl = PromptTemplate(TPL_GENERATION_BASE)

        # Get the length of the prompt template to truncate the prefix accordingly
        tpl_tokens = self.tokenizer(
            tpl.raw,
            return_attention_mask=False,
            add_special_tokens=False,
        )["input_ids"]

        prefix = prefix.rstrip("\n")
        prefix_truncated = truncate_content(
            self.tokenizer,
            prefix,
            max_length=max(self.model.MAX_MODEL_LEN - len(tpl_tokens), 0),
            truncation_side="left",
        )

        prompt_builder = PromptBuilder(prefix_truncated, file_name, lang_id=lang_id)
        prompt_builder.add_template(tpl)
        prompt = prompt_builder.build()

        return prompt
