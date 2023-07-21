import os
from codesuggestions.suggestions.processing.ops import LanguageId
from tree_sitter import Language, Node, Parser, Tree
from typing import Optional
from collections import Counter


class CodeParser:
    TREE_SITTER_LIB = "%s/tree-sitter-languages.so" % os.getenv("LIB_DIR", "/usr/lib")

    LANG_ID_TO_LANGUAGE = {
        LanguageId.C: "c",
        LanguageId.CPP: "cpp",
        LanguageId.CSHARP: "c_sharp",
        LanguageId.GO: "go",
        LanguageId.JAVA: "java",
        LanguageId.JS: "javascript",
        LanguageId.PHP: "php",
        LanguageId.PYTHON: "python",
        LanguageId.RUST: "rust",
        LanguageId.SCALA: "scala",
        LanguageId.TS: "typescript",
        # TODO Support Ruby. Not as straightforward since require uses a call node type.
    }

    LANGUAGES_TARGETS = {
        "imports": {
            LanguageId.C: "preproc_include",
            LanguageId.CPP: "preproc_include",
            LanguageId.CSHARP: "using_directive",
            LanguageId.GO: "import_declaration",
            LanguageId.JAVA: "import_declaration",
            LanguageId.JS: "import_statement",
            LanguageId.PHP: "namespace_use_declaration",
            LanguageId.PYTHON: "import_statement",
            LanguageId.RUST: "use_declaration",
            LanguageId.SCALA: "import_declaration",
            LanguageId.TS: 'import_statement',
        },
        "functions": {
            LanguageId.PYTHON: "function_definition",
            LanguageId.C: "function_definition",
        },
        "classes": {
            LanguageId.PYTHON: "class_definition",
        },
    }

    def __init__(self, lang_id: LanguageId):
        self.lang_id = lang_id
        self.parser = self.get_parser_for_lang_id(lang_id)
        if self.parser is None:
            raise ValueError(f"Language id not supported: {self.lang_id}")

    def extract_imports(self, code: str) -> list[str]:
        return self._extract_symbol(code, symbol="imports")

    def _extract_symbol(self, code: str, symbol: str) -> list[str]:
        nodes = []

        for node in self._each_node(code):
            if node.type == self.LANGUAGES_TARGETS[symbol][self.lang_id]:
                nodes.append(node)

        return [node.text.decode('utf-8', errors='ignore') for node in nodes]

    def count_symbols(self, code: str, target_symbols: Optional[set[str]]) -> dict:
        if target_symbols is None:
            target_symbols = self.LANGUAGES_TARGETS.keys()

        symbol_map = Counter()

        # TODO: bfs to traverse all nodes
        for node in self._each_node(code):
            for symbol in target_symbols:
                if symbol in self.LANGUAGES_TARGETS:
                    # does the node type match the lang-specific symbol name for this lang_id?
                    if node.type == self.LANGUAGES_TARGETS[symbol].get(self.lang_id, ""):
                        symbol_map.update([symbol])

        return symbol_map

    def get_parser_for_lang_id(self, lang_id: LanguageId) -> Optional[Parser]:
        if lang_id not in self.LANG_ID_TO_LANGUAGE:
            return None

        lang_def = self.LANG_ID_TO_LANGUAGE[lang_id]
        parser = Parser()
        language = Language(self.TREE_SITTER_LIB, lang_def)
        parser.set_language(language)

        return parser

    def _parse_code(self, code: str) -> Optional[Tree]:
        if self.parser is None:
            return None

        try:
            return self.parser.parse(bytes(code, "utf8"))
        except TypeError:
            return None

    def _each_node(self, code: str) -> Optional[Node]:
        tree = self._parse_code(code)

        if tree is None:
            return None

        root_node = tree.root_node

        if root_node is None:
            return None

        for node in root_node.children:
            yield node
