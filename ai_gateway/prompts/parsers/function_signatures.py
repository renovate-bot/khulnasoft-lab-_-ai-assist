from typing import Optional

from tree_sitter import Node

from ai_gateway.code_suggestions.processing.ops import LanguageId
from ai_gateway.prompts.parsers.base import BaseVisitor
from ai_gateway.prompts.parsers.mixins import RubyParserMixin

__all__ = [
    "BaseFunctionSignatureVisitor",
    "FunctionSignatureVisitorFactory",
]


class BaseFunctionSignatureVisitor(BaseVisitor):
    _FUNCTION_BODY_SYMBOL = "block"

    def __init__(self):
        self._signatures = []

    @property
    def function_signatures(self) -> list[str]:
        return self._signatures

    def _visit_node(self, node: Node):
        function_text = self._bytes_to_str(node.text)

        block_text = ""
        for child in node.children:
            if child.type == self._FUNCTION_BODY_SYMBOL:
                block_text = self._bytes_to_str(child.text)

        # remove the body of the function
        signature = function_text.replace(block_text, "")

        self._signatures.append(signature)


class CFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_definition"]
    _FUNCTION_BODY_SYMBOL = "compound_statement"


class CppFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_definition"]
    _FUNCTION_BODY_SYMBOL = "compound_statement"


# TODO: Not supporting c# for now because every function has to belong to a class in c#
# class CsharpFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
#     _TARGET_SYMBOLS = ["method_declaration"]


class GoFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_declaration"]


# TODO: Not supporting java for now because every function has to belong to a class in java
# class JavaFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
#     _TARGET_SYMBOLS = ["method_declaration"]


class JsFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_declaration"]
    _FUNCTION_BODY_SYMBOL = "statement_block"


class PhpFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_definition"]
    _FUNCTION_BODY_SYMBOL = "compound_statement"


class PythonFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_definition"]


class RubyFunctionSignatureVisitor(BaseFunctionSignatureVisitor, RubyParserMixin):
    _TARGET_SYMBOLS = ["method"]
    _FUNCTION_BODY_SYMBOL = "body_statement"

    def _visit_node(self, node: Node):
        function_text = self._bytes_to_str(node.text)

        block_text = ""
        for child in node.children:
            if child.type == self._FUNCTION_BODY_SYMBOL:
                block_text = self._bytes_to_str(child.text)

        # remove the body of the function
        signature = function_text.replace(block_text, "")

        # remove the last child node, which should be just "end"
        if node.children[-1].type == "end":
            signature = signature.replace("end", "")

        self._signatures.append(signature)


class RustFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_item"]


class ScalaFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_definition"]


class TsFunctionSignatureVisitor(BaseFunctionSignatureVisitor):
    _TARGET_SYMBOLS = ["function_declaration"]
    _FUNCTION_BODY_SYMBOL = "statement_block"


class FunctionSignatureVisitorFactory:
    _LANG_ID_VISITORS = {
        LanguageId.C: CFunctionSignatureVisitor,
        LanguageId.CPP: CppFunctionSignatureVisitor,
        LanguageId.GO: GoFunctionSignatureVisitor,
        LanguageId.JS: JsFunctionSignatureVisitor,
        LanguageId.PHP: PhpFunctionSignatureVisitor,
        LanguageId.PYTHON: PythonFunctionSignatureVisitor,
        LanguageId.RUBY: RubyFunctionSignatureVisitor,
        LanguageId.RUST: RustFunctionSignatureVisitor,
        LanguageId.SCALA: ScalaFunctionSignatureVisitor,
        LanguageId.TS: TsFunctionSignatureVisitor,
    }

    @staticmethod
    def from_language_id(lang_id: LanguageId) -> Optional[BaseFunctionSignatureVisitor]:
        if klass := FunctionSignatureVisitorFactory._LANG_ID_VISITORS.get(
            lang_id, None
        ):
            return klass()

        return None
