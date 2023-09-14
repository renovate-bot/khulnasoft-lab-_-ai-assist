from typing import List, Optional

from tree_sitter import Node

from ai_gateway.code_suggestions.processing.ops import LanguageId
from ai_gateway.prompts.parsers.base import BaseVisitor

__all__ = [
    "BaseContextVisitor",
    "PythonContextVisitor",
]


class BaseContextVisitor(BaseVisitor):
    # Descending order of importance.
    _TARGET_SYMBOLS = [
        "function_definition",
    ]

    def __init__(self, target_point: tuple[int, int]):
        self.visited_nodes = []
        self.target_point = target_point

    def _visit_node(self, node: Node):
        pass

    def visit(self, node: Node):
        if not node.is_named:
            return

        if self._is_point_within_rectangle(
            self.target_point, node.start_point, node.end_point
        ):
            self.visited_nodes.append(node)

    def _is_point_within_rectangle(
        self,
        target_point: tuple[int, int],
        rect_top_left: tuple[int, int],
        rect_bottom_right: tuple[int, int],
    ):
        target_row, _ = target_point
        start_row, _ = rect_top_left
        end_row, _ = rect_bottom_right

        return start_row <= target_row <= end_row

    def extract_most_relevant_context(
        self, priority_list: Optional[List[str]] = None
    ) -> Optional[Node]:
        if priority_list:
            priority_map = self._make_priority_map(priority_list)
        else:
            priority_map = self._make_priority_map(self._TARGET_SYMBOLS)
        curr_best = None
        curr_priority = -1
        for node in self.visited_nodes:
            if node.type in priority_map:
                node_priority = priority_map[node.type]
                if node_priority > curr_priority:
                    curr_priority = node_priority
                    curr_best = node
        return curr_best

    def _make_priority_map(self, importance_list):
        return {
            symbol: priority
            for priority, symbol in enumerate(reversed(importance_list))
        }


class PythonContextVisitor(BaseContextVisitor):
    _TARGET_SYMBOLS = [
        "class_definition",
        "function_definition",
        "module",
    ]


class JsContextVisitor(BaseContextVisitor):
    _TARGET_SYMBOLS = [
        "class_declaration",
        "lexical_declaration",
        "function_declaration",
        "generator_function_declaration",
    ]


class TsContextVisitor(BaseContextVisitor):
    _TARGET_SYMBOLS = [
        "class_declaration",
        "interface_declaration",
        "function_declaration",
        "generator_function_declaration",
        "call_expression",
        "program",
    ]


class ContextVisitorFactory:
    _LANG_ID_VISITORS = {
        # LanguageId.C: CCounterVisitor,
        # LanguageId.CPP: CppCounterVisitor,
        # LanguageId.CSHARP: CsharpCounterVisitor,
        # LanguageId.GO: GoCounterVisitor,
        # LanguageId.JAVA: JavaCounterVisitor,
        LanguageId.JS: JsContextVisitor,
        # LanguageId.PHP: PhpCounterVisitor,
        LanguageId.PYTHON: PythonContextVisitor,
        # LanguageId.RUBY: RubyCounterVisitor,
        # LanguageId.RUST: RustCounterVisitor,
        # LanguageId.SCALA: ScalaCounterVisitor,
        LanguageId.TS: TsContextVisitor,
    }

    @staticmethod
    def from_language_id(lang_id: LanguageId, point) -> Optional[BaseContextVisitor]:
        if klass := ContextVisitorFactory._LANG_ID_VISITORS.get(lang_id, None):
            return klass(point)

        return None
