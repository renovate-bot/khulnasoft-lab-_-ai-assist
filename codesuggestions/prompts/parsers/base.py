from abc import ABC, abstractmethod

from tree_sitter import Node

__all__ = [
    "BaseVisitor",
    "BaseCodeParser",
]


class BaseVisitor(ABC):
    _TARGET_SYMBOL = None

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
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
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
