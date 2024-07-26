from typing import Any, AsyncIterator, Union

from langchain_core.runnables import Runnable

from ai_gateway.models.base import ModelMetadata, SafetyAttributes
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)

__all__ = [
    "AgentModel",
]

AGENT = "agent"


class AgentModel(TextGenModelBase):
    def __init__(
        self,
        agent: Runnable,  # TODO: should be Agent, but SafetyAttributes complain about model_class_provider from TypeModelParams
    ):
        self.agent = agent
        self._metadata = ModelMetadata(
            name=agent.name,
            engine=AGENT,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(
        self,
        params: dict[str, Any],
        stream: bool = False,
    ) -> Union[TextGenModelOutput, AsyncIterator[TextGenModelChunk]]:

        if stream:
            return self._handle_stream(params)

        response = await self.agent.ainvoke(params)

        return TextGenModelOutput(
            text=response.content,
            # Give a high value, the model doesn't return scores.
            score=10**5,
            safety_attributes=SafetyAttributes(),
        )

    async def _handle_stream(
        self,
        params: dict,
    ) -> AsyncIterator[TextGenModelChunk]:
        async for chunk in self.agent.astream(params):
            yield TextGenModelChunk(text=chunk.content)
