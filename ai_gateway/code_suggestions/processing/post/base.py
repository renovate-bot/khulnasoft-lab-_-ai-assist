from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "PostProcessorBase",
]


class PostProcessorBase(ABC):
    @abstractmethod
    async def process(self, completion: str, **kwargs: Any) -> str:
        pass
