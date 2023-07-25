from collections import Counter
from typing import Optional

from tree_sitter import Node

from codesuggestions.prompts.parsers.base import BaseVisitor
from codesuggestions.suggestions.processing.ops import LanguageId

__all__ = [
    "BaseCounterVisitor",
    "CounterVisitorFactory",
]


class BaseCounterVisitor(BaseVisitor):
    _TARGET_SYMBOLS = set()

    def __init__(self):
        self._symbol_counter = Counter()

    @property
    def counts(self) -> dict:
        return dict(self._symbol_counter)

    def _visit_node(self, node: Node):
        self._symbol_counter.update([node.type])

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if node.type in self._TARGET_SYMBOLS:
            self._visit_node(node)


class CCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "preproc_include",
        "function_definition",
        "comment",
    }


class CppCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "preproc_include",
        "function_definition",
        "comment",
    }


class CsharpCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "using_directive",
        "class_declaration",
        "comment",
    }


class GoCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "import_declaration",
        "function_declaration",
        "comment",
    }


class JavaCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "import_declaration",
        "class_declaration",
        "line_comment",
        "block_comment",
    }


class JsCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "import_statement",
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "comment",
    }


class PhpCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "namespace_use_declaration",  # TODO: add support for require_once
        "function_definition",
        "class_declaration",
        "comment",
    }


class PythonCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "import_statement",
        "function_definition",
        "class_definition",
        "comment",
    }


class RustCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "use_declaration",
        "function_item",
        "line_comment",
        "block_comment",
    }


class ScalaCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "import_declaration",
        "function_definition",
        "class_definition",
        "comment",
    }


class TsCounterVisitor(BaseCounterVisitor):
    _TARGET_SYMBOLS = {
        "import_statement",
        "function_declaration",
        "class_declaration",
        "comment",
    }


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
        LanguageId.RUST: RustCounterVisitor,
        LanguageId.SCALA: ScalaCounterVisitor,
        LanguageId.TS: TsCounterVisitor,
    }

    @staticmethod
    def from_language_id(lang_id: LanguageId) -> Optional[BaseCounterVisitor]:
        if klass := CounterVisitorFactory._LANG_ID_VISITORS.get(lang_id, None):
            return klass()

        return None
