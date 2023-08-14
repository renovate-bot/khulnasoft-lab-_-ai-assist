from typing import Any

from codesuggestions.models import (
    PalmCodeGenBaseModel,
    VertexModelInternalError,
    VertexModelInvalidArgument,
)
from codesuggestions.suggestions.processing import (
    MetadataModel,
    ModelEngineBase,
    ModelEngineOutput,
)
from codesuggestions.suggestions.processing.ops import LanguageId

__all__ = [
    "ModelEngineGenerations",
]


class ModelEngineGenerations(ModelEngineBase):
    def __init__(self, model: PalmCodeGenBaseModel):
        self.model = model

    async def _generate(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        lang_id: LanguageId,
        **kwargs: Any,
    ) -> ModelEngineOutput:
        suggestion = ""
        model_metadata = MetadataModel(
            name=self.model.model_name, engine=self.model.model_engine
        )

        try:
            if res := await self.model.generate(prefix, suffix, **kwargs):
                suggestion = res.text
        except (VertexModelInvalidArgument, VertexModelInternalError):
            pass

        return ModelEngineOutput(
            text=suggestion,
            model=model_metadata,
            lang_id=lang_id,
        )
