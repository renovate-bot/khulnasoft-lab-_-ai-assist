import os
from typing import Optional

from tree_sitter import Language, Parser, Tree

from codesuggestions.prompts.parsers.base import BaseCodeParser, BaseVisitor
from codesuggestions.prompts.parsers.counters import CounterVisitorFactory
from codesuggestions.prompts.parsers.function_signatures import (
    FunctionSignatureVisitorFactory,
)
from codesuggestions.prompts.parsers.imports import ImportVisitorFactory
from codesuggestions.prompts.parsers.treetraversal import tree_dfs
from codesuggestions.suggestions.processing.ops import LanguageId


class CodeParser(BaseCodeParser):
    LANG_ID_TO_LANGUAGE = {
        LanguageId.C: "c",
        LanguageId.CPP: "cpp",
        LanguageId.CSHARP: "c_sharp",
        LanguageId.GO: "go",
        LanguageId.JAVA: "java",
        LanguageId.JS: "javascript",
        LanguageId.PHP: "php",
        LanguageId.PYTHON: "python",
        LanguageId.RUBY: "ruby",
        LanguageId.RUST: "rust",
        LanguageId.SCALA: "scala",
        LanguageId.TS: "typescript",
    }

    def __init__(self, tree: Tree, lang_id: LanguageId):
        self.tree = tree
        self.lang_id = lang_id

    def imports(self) -> list[str]:
        visitor = ImportVisitorFactory.from_language_id(self.lang_id)
        if visitor is None:
            return []

        self._visit_nodes(visitor)
        imports = visitor.imports

        return imports

    def function_signatures(self) -> list[str]:
        visitor = FunctionSignatureVisitorFactory.from_language_id(self.lang_id)
        if visitor is None:
            return []

        self._visit_nodes(visitor)
        function_signatures = visitor.function_signatures

        return function_signatures

    def count_symbols(self) -> dict:
        visitor = CounterVisitorFactory.from_language_id(self.lang_id)
        if visitor is None:
            return []

        self._visit_nodes(visitor)
        counts = visitor.counts

        return counts

    def _visit_nodes(self, visitor: BaseVisitor):
        tree_dfs(self.tree, visitor)

    @classmethod
    def from_language_id(
        cls,
        content: str,
        lang_id: LanguageId,
        lib_path: Optional[str] = None,
    ):
        if lib_path is None:
            lib_path = "%s/tree-sitter-languages.so" % os.getenv("LIB_DIR", "/usr/lib")

        lang_def = cls.LANG_ID_TO_LANGUAGE.get(lang_id, None)
        if lang_def is None:
            raise ValueError(f"Unsupported language: {lang_id}")

        try:
            parser = Parser()
            parser.set_language(Language(lib_path, lang_def))
            tree = parser.parse(bytes(content, "utf8"))
        except TypeError as ex:
            raise ValueError(f"Unsupported code content: {str(ex)}")

        return cls(tree, lang_id)
