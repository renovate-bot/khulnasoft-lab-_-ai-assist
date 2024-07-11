from abc import ABC, abstractmethod
from typing import AsyncIterator, NamedTuple, Optional

from ai_gateway.models.base import (
    ModelBase,
    SafetyAttributes,
    TokensConsumptionMetadata,
)

__all__ = ["TextGenModelOutput", "TextGenModelChunk", "TextGenModelBase"]


class TextGenModelOutput(NamedTuple):
    text: str
    score: Optional[float] = None
    safety_attributes: Optional[SafetyAttributes] = None
    metadata: Optional[TokensConsumptionMetadata] = None


class TextGenModelChunk(NamedTuple):
    text: str


class TextGenModelBase(ModelBase, ABC):
    @abstractmethod
    async def generate(
        self,
        prefix: str,
        suffix: str,
        stream: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 16,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> (
        TextGenModelOutput | list[TextGenModelOutput] | AsyncIterator[TextGenModelChunk]
    ):
        pass
