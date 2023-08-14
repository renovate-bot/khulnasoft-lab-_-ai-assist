import pytest
from tree_sitter import Node

from codesuggestions.prompts.parsers import CodeParser
from codesuggestions.prompts.parsers.context_extractors import BaseContextVisitor
from codesuggestions.prompts.parsers.treetraversal import tree_dfs
from codesuggestions.suggestions.processing.ops import (
    LanguageId,
    point_to_position,
    split_on_point,
)

PYTHON_PREFIX_SAMPLE = """
from abc import ABC
from abc import abstractmethod

from tree_sitter import Node


class BaseVisitor(ABC):
    _TARGET_SYMBOL = None

    @abstractmethod
    def _visit_node(self, node: Node):
        pass

    @property
    def stop_earlier(self) -> bool:
        return False

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
            self._visit_node("""

PYTHON_SUFFIX_SAMPLE = """node)


class BaseCodeParser(ABC):
    @abstractmethod
    def count_symbols(self) -> dict:
        pass

    @abstractmethod
    def imports(self) -> list[str]:
        pass

"""

_PYTHON_PREFIX_EXPECTED_FUNCTION_DEFINITION_CONTEXT = """
def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
            self._visit_node(
"""

_PYTHON_PREFIX_EXPECTED_CLASS_DEFINITION_CONTEXT = """
class BaseVisitor(ABC):
    _TARGET_SYMBOL = None

    @abstractmethod
    def _visit_node(self, node: Node):
        pass

    @property
    def stop_earlier(self) -> bool:
        return False

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
            self._visit_node(
"""


@pytest.mark.parametrize(
    (
        "lang_id",
        "source_code",
        "target_point",
        "expected_node_count",
        "expected_context",
        "priority_list",
    ),
    [
        (
            LanguageId.PYTHON,
            PYTHON_PREFIX_SAMPLE,
            (21, 29),
            8,
            _PYTHON_PREFIX_EXPECTED_FUNCTION_DEFINITION_CONTEXT,
            ["function_definition"],
        ),
        (
            LanguageId.PYTHON,
            PYTHON_PREFIX_SAMPLE,
            (21, 29),
            8,
            _PYTHON_PREFIX_EXPECTED_CLASS_DEFINITION_CONTEXT,
            ["class_definition"],
        ),
        (
            LanguageId.PYTHON,
            PYTHON_PREFIX_SAMPLE + PYTHON_SUFFIX_SAMPLE,
            (21, 29),
            10,
            """def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOL and node.type == self._TARGET_SYMBOL:
            self._visit_node(node)
""",
            ["function_definition"],
        ),
    ],
)
def test_base_context_visitor(
    lang_id: LanguageId,
    source_code: str,
    target_point: tuple[int, int],
    expected_node_count: int,
    expected_context: str,
    priority_list: list[str],
):
    parser = CodeParser.from_language_id(source_code, lang_id)
    visitor = BaseContextVisitor(target_point)
    tree_dfs(parser.tree, visitor)

    context_node = visitor.extract_most_relevant_context(priority_list=priority_list)
    assert context_node is not None

    actual_context = visitor._bytes_to_str(context_node.text)

    assert len(visitor.visited_nodes) == expected_node_count
    assert actual_context.strip() == expected_context.strip()


PYTHON_SAMPLE_TWO_FUNCTIONS = """
def sum(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""

PYTHON_SAMPLE_FUNCTION_WITHIN_FUNCTION = """
import os

def i_want_to_sum(a, b):
    def sum(a, b):
        return a + b
    return sum(a, b)
"""

PYTHON_SAMPLE_TWO_CLASSES = """
from abc import ABC, abstractmethod

from tree_sitter import Node

__all__ = [
    "BaseVisitor",
    "BaseCodeParser",
]


class BaseVisitor(ABC):
    _TARGET_SYMBOLS = []

    @abstractmethod
    def _visit_node(self, node: Node):
        pass

    @property
    def stop_tree_traversal(self) -> bool:
        return False

    @property
    def stop_node_traversal(self) -> bool:
        return False

    def visit(self, node: Node):
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOLS and node.type in self._TARGET_SYMBOLS:
            self._visit_node(node)

    def _bytes_to_str(self, data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")


class BaseCodeParser(ABC):
    @abstractmethod
    def count_symbols(self) -> dict:
        pass

    @abstractmethod
    def imports(self) -> list[str]:
        pass
"""

PYTHON_SAMPLE_CLASS_WITHIN_CLASS = """
class SuggestionsResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    class Model(BaseModel):
        engine: str
        name: str
        lang: str

    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]
"""


@pytest.mark.parametrize(
    (
        "lang_id",
        "source_code",
        "target_point",
        "expected_prefix",
        "expected_suffix",
    ),
    [
        # TODO: Add a test to sweep the full range of the context rectangle
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (0, 14),
            "def sum(a, b):",
            "\n    return a + b",
        ),
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (0, 13),
            "def sum(a, b)",
            ":\n    return a + b",
        ),
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (0, 12),
            "def sum(a, b",
            "):\n    return a + b",
        ),
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (0, 11),
            "def sum(a, ",
            "b):\n    return a + b",
        ),
        (  # Test context at function level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_FUNCTION_WITHIN_FUNCTION[1:],
            (2, 20),
            "import os\n\ndef i_want_to_sum(a,",
            " b):\n    def sum(a, b):\n        return a + b\n    return sum(a, b)",
        ),
        (  # Test context at module level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_FUNCTIONS[1:],
            (2, 0),
            "def sum(a, b):\n    return a + b\n",
            "\ndef subtract(a, b):\n    return a - b\n",
        ),
        (  # Test context at class level
            LanguageId.PYTHON,
            PYTHON_SAMPLE_TWO_CLASSES[1:],
            (25, 32),
            # fmt: off
            PYTHON_SAMPLE_TWO_CLASSES[1:438], # last line:'     def visit(self, node: Node):'
"""
        # use self instead of the class name to access the overridden attribute
        if self._TARGET_SYMBOLS and node.type in self._TARGET_SYMBOLS:
            self._visit_node(node)

    def _bytes_to_str(self, data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")
"""[:-1],
            # fmt: on
        ),
        (  # Test context at class within class, cursor within nested the class
            LanguageId.PYTHON,
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:],
            (4, 21),
            # fmt: off
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:130],
""": str = "length"

    class Model(BaseModel):
        engine: str
        name: str
        lang: str

    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]
"""[:-1],
            # fmt: on
        ),
        (  # Test context at class within class, cursor within the 2nd nested class
            LanguageId.PYTHON,
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:],
            (7, 15),
            # fmt: off
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:191],
""" str
        name: str
        lang: str

    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]
"""[:-1],
            # fmt: on
        ),
        (  # Test context at class within class, cursor within outer class
            LanguageId.PYTHON,
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:],
            (11, 0),
            # fmt: off
            PYTHON_SAMPLE_CLASS_WITHIN_CLASS[1:233],
"""
    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]
"""[1:-1],
            # fmt: on
        ),
    ],
)
def test_suffix_near_cursor(
    lang_id: LanguageId,
    source_code: str,
    target_point: tuple[int, int],
    expected_prefix: str,
    expected_suffix: str,
):
    parser = CodeParser.from_language_id(source_code, lang_id)
    actual_prefix, _ = split_on_point(source_code, target_point)

    print(f"{target_point=}")
    print("-----------------------")
    print("source_code:")
    print("-----------------------")
    pos = point_to_position(source_code, target_point)
    print(_highlight_position(pos, source_code))

    actual_truncated_suffix = parser.suffix_near_cursor(target_point)

    print("-----------------------")
    print("Prefix")
    print("-----------------------")
    print(repr(actual_prefix))
    print(repr(expected_prefix))

    print("-----------------------")
    print("Suffix")
    print("-----------------------")
    print(repr(actual_truncated_suffix))
    print(repr(expected_suffix))

    assert actual_prefix == expected_prefix
    assert actual_truncated_suffix == expected_suffix


def _highlight_position(pos, mystring):
    # fix this quadratic loop
    text_highlight = ""
    for i, x in enumerate(mystring):
        if i == pos:
            text_highlight += f"\033[44;33m{x}\033[m"
        else:
            text_highlight += x
    return text_highlight
