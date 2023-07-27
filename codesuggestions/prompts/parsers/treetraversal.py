from collections import deque
from typing import Callable, Union

from tree_sitter import Node, Tree

from codesuggestions.prompts.parsers.base import BaseVisitor

__all__ = [
    "tree_bfs",
]


def tree_bfs(node: Node, visitor: Union[Callable, BaseVisitor], max_depth: int = 0):
    depth = 0
    q1 = deque([*node.children])
    q2 = deque()

    while q1:
        node = q1.popleft()
        if isinstance(visitor, Callable):
            visitor(depth, node)
        else:
            visitor.visit(node)

        # add the next level nodes to the queue
        q2.extend(node.children)

        # no more nodes to process, break the execution
        if len(q1) == 0 and len(q2) == 0:
            break

        # no more nodes to process at this level, check the next one
        if len(q1) == 0:
            depth += 1

            # if max_depth is lower than 0,
            # iterate until the last visited node in the tree
            if 0 <= max_depth < depth:
                break

            q1.extend(q2)
            q2.clear()


def tree_dfs(tree: Tree, visitor: BaseVisitor):
    cursor = tree.walk()
    has_next = True

    while has_next:
        current_node = cursor.node

        if visitor.stop_earlier:
            break

        visitor.visit(current_node)
        has_next = cursor.goto_first_child()

        if not has_next:
            has_next = cursor.goto_next_sibling()

        while not has_next and cursor.goto_parent():
            has_next = cursor.goto_next_sibling()
