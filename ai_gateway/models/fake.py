from typing import Optional

from ai_gateway.models import TextGenBaseModel, TextGenModelOutput

__all__ = [
    "FakePalmTextGenModel",
]


class FakePalmTextGenModel(TextGenBaseModel):
    @property
    def model_name(self) -> str:
        return "fake-palm-model"

    @property
    def model_engine(self) -> str:
        return "fake-palm-engine"

    async def generate(
        self,
        prompt: str,
        suffix: str,
        temperature: float = 0.0,
        max_output_tokens: int = 0,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> Optional[TextGenModelOutput]:
        return TextGenModelOutput(text="fake code suggestion from PaLM Text", score=0)
