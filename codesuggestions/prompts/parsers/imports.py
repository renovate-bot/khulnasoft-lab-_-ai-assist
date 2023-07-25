from typing import Optional

from tree_sitter import Node

from codesuggestions.prompts.parsers.base import BaseVisitor

__all__ = [
    "BaseImportVisitor",
    "ImportVisitorFactory",
]

from codesuggestions.suggestions.processing.ops import LanguageId


class BaseImportVisitor(BaseVisitor):
    def __init__(self):
        self._imports = []

    @property
    def imports(self) -> list[str]:
        return self._imports

    def _visit_node(self, node: Node):
        self._imports.append(node.text.decode('utf-8', errors='ignore'))


class CImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "preproc_include"


class CppImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "preproc_include"


class CsharpImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "using_directive"


class GoImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "import_declaration"


class JavaImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "import_declaration"


class JsImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "import_statement"


class PhpImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "namespace_use_declaration"


class PythonImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "import_statement"


class RustImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "use_declaration"


class ScalaImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "import_declaration"


class TsImportVisitor(BaseImportVisitor):
    _TARGET_SYMBOL = "import_statement"


class ImportVisitorFactory:
    _LANG_ID_VISITORS = {
        LanguageId.C: CImportVisitor,
        LanguageId.CPP: CppImportVisitor,
        LanguageId.CSHARP: CsharpImportVisitor,
        LanguageId.GO: GoImportVisitor,
        LanguageId.JAVA: JavaImportVisitor,
        LanguageId.JS: JsImportVisitor,
        LanguageId.PHP: PhpImportVisitor,
        LanguageId.PYTHON: PythonImportVisitor,
        LanguageId.RUST: RustImportVisitor,
        LanguageId.SCALA: ScalaImportVisitor,
        LanguageId.TS: TsImportVisitor,
    }

    @staticmethod
    def from_language_id(lang_id: LanguageId) -> Optional[BaseImportVisitor]:
        if klass := ImportVisitorFactory._LANG_ID_VISITORS.get(lang_id, None):
            return klass()

        return None
