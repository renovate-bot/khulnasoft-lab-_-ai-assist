from typing import Optional

from tree_sitter import Node

from ai_gateway.code_suggestions.processing.ops import LanguageId
from ai_gateway.prompts.parsers.base import BaseVisitor

__all__ = [
    "BaseCommentVisitor",
    "CommentVisitorFactory",
]


class BaseCommentVisitor(BaseVisitor):
    def __init__(self):
        self._stop_tree_traversal = False
        self._stop_node_traversal = False
        self._comments_only = True

    @property
    def stop_tree_traversal(self) -> bool:
        return self._stop_tree_traversal

    @property
    def stop_node_traversal(self) -> bool:
        return self._stop_node_traversal

    @property
    def comments_only(self) -> bool:
        return self._comments_only

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOLS and node.type not in self._TARGET_SYMBOLS:
            self._comments_only = False
            self._stop_node_traversal = True
            self._stop_tree_traversal = True

    def _visit_node(self, node: Node):
        return super()._visit_node(node)


class CCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "translation_unit",
    ]


class CppCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "translation_unit",
    ]


class CSharpCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "compilation_unit",
    ]


class GoCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "source_file",
    ]


class JavaCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "line_comment",
        "block_comment",
        "program",
    ]


class JsCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "program",
    ]


# The PHP grammar only recognises comments if they are surrounded by 'php_tags';
# This is not ideal for our use case
# class PhpCommentVisitor(BaseCommentVisitor):
#     _TARGET_SYMBOLS = [
#         "comment",
#         "program",
#         "php_tag",
#         "text_interpolation",
#         "?>",
#     ]


class PythonCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "module",
    ]


class RubyCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "program",
    ]


class RustCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "line_comment",
        "source_file",
    ]


class ScalaCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "//",
        "block_comment",
        "/*",
        "*/",
        "compilation_unit",
    ]


class TsCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "comment",
        "program",
    ]


class KotlinCommentVisitor(BaseCommentVisitor):
    _TARGET_SYMBOLS = [
        "line_comment",
        "multiline_comment",
        "source_file",
    ]


class CommentVisitorFactory:
    _LANG_ID_VISITORS = {
        LanguageId.C: CCommentVisitor,
        LanguageId.CPP: CppCommentVisitor,
        LanguageId.CSHARP: CSharpCommentVisitor,
        LanguageId.GO: GoCommentVisitor,
        LanguageId.JAVA: JavaCommentVisitor,
        LanguageId.JS: JsCommentVisitor,
        LanguageId.PYTHON: PythonCommentVisitor,
        LanguageId.RUBY: RubyCommentVisitor,
        LanguageId.RUST: RustCommentVisitor,
        LanguageId.SCALA: ScalaCommentVisitor,
        LanguageId.TS: TsCommentVisitor,
        LanguageId.KOTLIN: KotlinCommentVisitor,
    }

    @staticmethod
    def from_language_id(lang_id: LanguageId) -> Optional[BaseCommentVisitor]:
        if klass := CommentVisitorFactory._LANG_ID_VISITORS.get(lang_id, None):
            return klass()

        return None
