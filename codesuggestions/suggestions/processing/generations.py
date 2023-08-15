from typing import Any

from codesuggestions.models import VertexModelInternalError, VertexModelInvalidArgument
from codesuggestions.suggestions.processing.base import (
    MetadataModel,
    MetadataPromptBuilder,
    ModelEngineBase,
    ModelEngineOutput,
    Prompt,
    PromptBuilderBase,
)
from codesuggestions.suggestions.processing.ops import LanguageId, truncate_content

__all__ = [
    "ModelEngineGenerations",
]


class _PromptBuilder(PromptBuilderBase):
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
    async def _generate(
        self,
        prefix: str,
        _suffix: str,
        file_name: str,
        lang_id: LanguageId,
        **kwargs: Any,
    ) -> ModelEngineOutput:
        model_metadata = MetadataModel(
            name=self.model.model_name, engine=self.model.model_engine
        )

        prompt = self._build_prompt(prefix, lang_id)
        with self.instrumentator.watch(prompt) as watch_container:
            try:
                if res := await self.model.generate(prompt.prefix, "", **kwargs):
                    watch_container.register_model_output_length(res.text)
                    watch_container.register_model_score(res.score)

                    return ModelEngineOutput(
                        text=res.text,
                        model=model_metadata,
                        lang_id=lang_id,
                        metadata=prompt.metadata,
                    )

            except (VertexModelInvalidArgument, VertexModelInternalError) as ex:
                watch_container.register_model_exception(str(ex), ex.code)

        return ModelEngineOutput(text="", model=model_metadata)

    def _build_prompt(self, prefix: str, lang_id: LanguageId):
        prefix_truncated = truncate_content(
            self.tokenizer,
            prefix,
            max_length=self.model.MAX_MODEL_LEN,
            truncation_side="left",
        )

        prompt_builder = _PromptBuilder(prefix_truncated, lang_id=lang_id)
        prompt = prompt_builder.build()

        return prompt
