from abc import ABC, abstractmethod
from typing import NamedTuple

from tree_sitter import Node

__all__ = [
    "Point",
    "CodeContext",
    "BaseVisitor",
    "BaseCodeParser",
]

Point = tuple[int, int]


class CodeContext(NamedTuple):
    text: str
    start: Point
    end: Point

    @classmethod
    def from_node(cls, node: Node):
        return cls(
            text=node.text.decode("utf-8", errors="ignore"),
            start=node.start_point,
            end=node.end_point,
        )


class BaseVisitor(ABC):
    _TARGET_SYMBOLS = []

    @abstractmethod
    def _visit_node(self, node: Node):
        pass

    @property
    def stop_tree_traversal(self) -> bool:
        return False

    @property
    def stop_node_traversal(self) -> bool:
        return False

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOLS and node.type in self._TARGET_SYMBOLS:
            self._visit_node(node)

    def _bytes_to_str(self, data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")


class BaseCodeParser(ABC):
    @abstractmethod
    def count_symbols(self) -> dict:
        pass

    @abstractmethod
    def imports(self) -> list[str]:
        pass
