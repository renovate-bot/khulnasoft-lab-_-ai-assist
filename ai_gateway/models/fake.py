from typing import Any, Optional

from ai_gateway.models.base import (
    ModelMetadata,
    SafetyAttributes,
    TextGenBaseModel,
    TextGenModelOutput,
)

__all__ = [
    "FakePalmTextGenModel",
]


class FakePalmTextGenModel(TextGenBaseModel):
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(name="fake-palm-model", engine="fake-palm-engine")

    async def generate(
        self,
        prompt: str,
        suffix: str,
        stream: bool = False,
        temperature: float = 0.0,
        max_output_tokens: int = 0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> Optional[TextGenModelOutput]:
        return TextGenModelOutput(
            text="fake code suggestion from PaLM Text",
            score=0,
            safety_attributes=SafetyAttributes(),
        )
