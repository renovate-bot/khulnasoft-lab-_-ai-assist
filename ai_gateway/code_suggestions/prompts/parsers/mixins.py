from tree_sitter import Node

__all__ = ["RubyParserMixin"]


class RubyParserMixin:
    def is_import(self, node: Node) -> bool:
        if len(node.children) != 2:
            return False

        first, second = node.children
        first_text = first.text.decode("utf-8", errors="ignore")

        return (
            first_text in {"require", "require_relative"}
            and first.type == "identifier"
            and second.type == "argument_list"
        )
