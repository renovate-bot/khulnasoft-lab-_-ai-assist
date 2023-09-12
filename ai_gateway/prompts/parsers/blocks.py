from typing import Optional

from tree_sitter import Node

from ai_gateway.prompts.parsers.base import BaseVisitor, Point

__all__ = ["MinAllowedBlockVisitor"]


class MinAllowedBlockVisitor(BaseVisitor):
    def __init__(self, target_point: Point, min_block_size: int = 2):
        self.target_point = target_point
        self.min_block_size = min_block_size
        self.visited_nodes = []

    def _visit_node(self, node: Node):
        if self._is_block_candidate(node) and self._is_point_included(node):
            self.visited_nodes.append(node)

    def _is_point_included(self, node: Node) -> bool:
        target_row, target_col = self.target_point
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point

        return start_row <= target_row <= end_row and start_col <= target_col <= end_col

    def _is_block_candidate(self, node: Node) -> bool:
        if node.end_point[0] - node.start_point[0] + 1 >= self.min_block_size:
            return True

    @property
    def block(self) -> Optional[Node]:
        self.visited_nodes.sort(key=lambda node: node.end_point, reverse=True)
        return self.visited_nodes[-1] if self.visited_nodes else None

    def visit(self, node: Node):
        # override the inherited method to visit all nodes
        self._visit_node(node)
