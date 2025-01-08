from typing import Any, AsyncIterator, Optional

from ai_gateway.models.base import ModelMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.prompts.base import Prompt
from ai_gateway.safety_attributes import SafetyAttributes

__all__ = [
    "AgentModel",
]

AGENT = "agent"


class AgentModel(TextGenModelBase):
    def __init__(
        self,
        prompt: Prompt,
    ):
        self.prompt = prompt
        self._metadata = ModelMetadata(
            name=prompt.name,
            engine=AGENT,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(  # type: ignore[override]
        self,
        params: dict[str, Any],
        stream: bool = False,
    ) -> (
        TextGenModelOutput | list[TextGenModelOutput] | AsyncIterator[TextGenModelChunk]
    ):
        if stream:
            return self._handle_stream(params)

        response = await self.prompt.ainvoke(params)

        response_content = self._format_response_content(response.content)

        return TextGenModelOutput(
            text=response_content,
            # Give a high value, the model doesn't return scores.
            score=10**5,
            safety_attributes=SafetyAttributes(),
        )

    async def _handle_stream(
        self,
        params: dict,
    ) -> AsyncIterator[TextGenModelChunk]:
        async for chunk in self.prompt.astream(params):
            chunk_content = self._format_response_content(chunk.content)
            yield TextGenModelChunk(text=chunk_content)

    def _format_response_content(self, text: Optional[str]) -> Optional[str]:
        if text and self.prompt.model_name == "mixtral":
            return text.replace("\\_", "_")

        return text
