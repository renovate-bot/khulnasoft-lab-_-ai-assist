from collections import Counter
from typing import Optional

from tree_sitter import Node

from ai_gateway.code_suggestions.processing.ops import LanguageId
from ai_gateway.prompts.parsers.base import BaseVisitor
from ai_gateway.prompts.parsers.mixins import RubyParserMixin

__all__ = [
    "BaseCounterVisitor",
    "CounterVisitorFactory",
]


class BaseCounterVisitor(BaseVisitor):
    def __init__(self):
        self._symbol_counter = Counter()

    @property
    def counts(self) -> dict:
        return dict(self._symbol_counter)

    def _visit_node(self, node: Node):
        self._symbol_counter.update([node.type])


class CCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "preproc_include",
        "function_definition",
        "comment",
    ]


class CppCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "preproc_include",
        "function_definition",
        "comment",
    ]


class CsharpCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "using_directive",
        "class_declaration",
        "comment",
    ]


class GoCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "import_declaration",
        "function_declaration",
        "comment",
    ]


class JavaCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "import_declaration",
        "class_declaration",
        "line_comment",
        "block_comment",
    ]


class JsCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "import_statement",
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "comment",
    ]


class PhpCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "namespace_use_declaration",  # TODO: add support for require_once
        "function_definition",
        "class_declaration",
        "comment",
    ]


class PythonCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "import_statement",
        "function_definition",
        "class_definition",
        "comment",
    ]


class RubyCounterVisitor(BaseCounterVisitor, RubyParserMixin):
    _TARGET_SYMBOLS = [
        "comment",
        "call",
        "module",
        "class",
    ]

    def _visit_node(self, node: Node):
        if node.type == "call":
            if self.is_import(node):
                # Remap call to require since call is too generic
                self._symbol_counter.update(["require"])
        else:
            # In the Ruby grammar, module and class definitions get two nodes, one
            # with children and one without. For example:
            #
            # module Foo
            #   def initalize(self):
            #   end
            # end
            #
            # The parser returns:
            #
            # 1. A `module` node type for the entire `module` definition.
            # 2. Another `module` node type for just the `module Foo` part, with no children.
            if (
                node.type == "comment"
                or node.type in {"module", "class"}
                and len(node.children) > 0
            ):
                self._symbol_counter.update([node.type])


class RustCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "use_declaration",
        "function_item",
        "line_comment",
        "block_comment",
    ]


class ScalaCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "import_declaration",
        "function_definition",
        "class_definition",
        "comment",
    ]


class TsCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "import_statement",
        "function_declaration",
        "class_declaration",
        "comment",
    ]


class KotlinCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = [
        "import_header",
        "function_declaration",
        "class_declaration",
        "line_comment",
        "multiline_comment",
    ]


class CounterVisitorFactory:
    _LANG_ID_VISITORS = {
        LanguageId.C: CCounterVisitor,
        LanguageId.CPP: CppCounterVisitor,
        LanguageId.CSHARP: CsharpCounterVisitor,
        LanguageId.GO: GoCounterVisitor,
        LanguageId.JAVA: JavaCounterVisitor,
        LanguageId.JS: JsCounterVisitor,
        LanguageId.PHP: PhpCounterVisitor,
        LanguageId.PYTHON: PythonCounterVisitor,
        LanguageId.RUBY: RubyCounterVisitor,
        LanguageId.RUST: RustCounterVisitor,
        LanguageId.SCALA: ScalaCounterVisitor,
        LanguageId.TS: TsCounterVisitor,
        LanguageId.KOTLIN: KotlinCounterVisitor,
    }

    @staticmethod
    def from_language_id(lang_id: LanguageId) -> Optional[BaseCounterVisitor]:
        if klass := CounterVisitorFactory._LANG_ID_VISITORS.get(lang_id, None):
            return klass()

        return None
