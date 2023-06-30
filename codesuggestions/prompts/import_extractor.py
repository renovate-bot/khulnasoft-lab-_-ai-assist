from codesuggestions.suggestions.processing.base import LanguageId
from tree_sitter import Language, Parser
from typing import NamedTuple


class ImportExtractor:
    class NodeType(NamedTuple):
        language: Language
        node_type: str

    TREE_SITTER_LIB = 'lib/tree-sitter-languages.so'

    TREE_SITTER_LANGUAGES = {
        LanguageId.C: NodeType(Language(TREE_SITTER_LIB, 'c'), 'preproc_include'),
        LanguageId.CPP: NodeType(Language(TREE_SITTER_LIB, 'cpp'), 'preproc_include'),
        LanguageId.CSHARP: NodeType(Language(TREE_SITTER_LIB, 'c_sharp'), 'using_directive'),
        LanguageId.GO: NodeType(Language(TREE_SITTER_LIB, 'go'), 'import_declaration'),
        LanguageId.JAVA: NodeType(Language(TREE_SITTER_LIB, 'java'), 'import_declaration'),
        LanguageId.JS: NodeType(Language(TREE_SITTER_LIB, 'javascript'), 'import_statement'),
        LanguageId.PHP: NodeType(Language(TREE_SITTER_LIB, 'php'), 'namespace_use_declaration'),
        LanguageId.PYTHON: NodeType(Language(TREE_SITTER_LIB, 'python'), 'import_statement'),
        LanguageId.RUST: NodeType(Language(TREE_SITTER_LIB, 'rust'), 'use_declaration'),
        LanguageId.SCALA: NodeType(Language(TREE_SITTER_LIB, 'scala'), 'import_declaration'),
        LanguageId.TS: NodeType(Language(TREE_SITTER_LIB, 'typescript'), 'import_statement'),
        # TODO Support Ruby. Not as straightforward since require uses a call node type.
    }

    def __init__(self, lang_id: LanguageId):
        self.parser = None

        if lang_id in self.TREE_SITTER_LANGUAGES:
            lang_def = self.TREE_SITTER_LANGUAGES[lang_id]
            self.node_type = lang_def.node_type
            self.language = lang_def.language

            self.parser = Parser()
            self.parser.set_language(self.language)

    def extract_imports(self, code: str) -> list[str]:
        if self.parser is None:
            return None

        try:
            tree = self.parser.parse(bytes(code, "utf8"))
        except TypeError:
            return None

        root_node = tree.root_node

        if root_node is None:
            return None

        import_nodes = []

        for node in root_node.children:
            if node.type == self.node_type:
                import_nodes.append(node)

        return [node.text.decode('utf-8', errors='ignore') for node in import_nodes]
