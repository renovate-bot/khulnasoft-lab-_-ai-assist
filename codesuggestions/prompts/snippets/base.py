from abc import ABC, abstractmethod
from typing import Iterable, NamedTuple

__all__ = [
    "CodeSnippet",
    "BaseCodeSnippetsIterator",
]


class CodeSnippet(NamedTuple):
    text: str
    start_pos: tuple[int, int]
    end_pos: tuple[int, int]


class BaseCodeSnippetsIterator(ABC):
    def __init__(self, content: str):
        self.content = content

    def __iter__(self):
        yield from self._next_snippet()

    @abstractmethod
    def _next_snippet(self) -> Iterable[CodeSnippet]:
        pass
