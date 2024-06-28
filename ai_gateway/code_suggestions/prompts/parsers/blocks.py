from typing import List, Optional

from tree_sitter import Node

from ai_gateway.code_suggestions.prompts.parsers.base import BaseVisitor, Point

__all__ = ["MinAllowedBlockVisitor", "ErrorBlocksVisitor"]


class MinAllowedBlockVisitor(BaseVisitor):
    def __init__(self, target_point: Point, min_block_size: int = 2):
        self.target_point = target_point
        self.min_block_size = min_block_size
        self.visited_nodes: List[Node] = []

    def _visit_node(self, node: Node):
        if self._is_block_candidate(node) and self._is_point_included(node):
            self.visited_nodes.append(node)

    def _is_point_included(self, node: Node) -> bool:
        target_row, target_col = self.target_point
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point

        if start_row == target_row:
            return start_col <= target_col
        if end_row == target_row:
            return end_col >= target_col

        return start_row < target_row < end_row

    def _is_block_candidate(self, node: Node) -> bool:
        if node.end_point[0] - node.start_point[0] + 1 >= self.min_block_size:
            return True
        return False

    @property
    def block(self) -> Optional[Node]:
        self.visited_nodes.sort(key=lambda node: node.end_point, reverse=True)
        return self.visited_nodes[-1] if self.visited_nodes else None

    def visit(self, node: Node):
        # override the inherited method to visit all nodes
        self._visit_node(node)


class ErrorBlocksVisitor(BaseVisitor):
    def __init__(self):
        self.error_nodes = []

    def _visit_node(self, target_node: Node):
        # Include only low-level errors that do not contain other error nodes.
        # We assume that the visitor called with the DFS algorithm.
        # Consider updating the logic by building a binary tree of visited nodes if the support of BFS is required.
        nodes = [target_node]
        for node in self.error_nodes:
            if (
                node.start_point <= target_node.start_point
                and target_node.end_point <= node.end_point
            ):
                continue
            nodes.append(node)

        self.error_nodes = nodes

    @property
    def errors(self) -> list[Node]:
        return self.error_nodes

    def visit(self, node: Node):
        # override the inherited method to rely on the error property
        # instead of the node name
        if node.has_error:
            self._visit_node(node)
