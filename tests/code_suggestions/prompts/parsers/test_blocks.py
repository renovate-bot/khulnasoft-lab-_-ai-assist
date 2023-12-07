import pytest

from ai_gateway.code_suggestions.processing.ops import LanguageId
from ai_gateway.prompts.parsers import (
    CodeParser,
    ErrorBlocksVisitor,
    MinAllowedBlockVisitor,
    Point,
)
from ai_gateway.prompts.parsers.treetraversal import tree_dfs

PYTHON_SAMPLE = """
class BaseContextVisitor(BaseVisitor):
    _TARGET_SYMBOLS = [
        "function_definition",
    ]

    def __init__(self, target_point: tuple[int, int]):
        self.visited_nodes = []
        self.target_point = target_point

    def _visit_node(self, node: Node):
        pass

    def visit(self, node: Node):
        if self._is_point_within_rectangle(
            self.target_point, node.start_point, node.end_point
        ):
            self.visited_nodes.append(node)

    def extract_most_relevant_context(
        self, priority_list: Optional[List[str]] = None
    ) -> Optional[Node]:
        if priority_list:
            priority_map = self._make_priority_map(priority_list)
        else:
            priority_map = self._make_priority_map(self._TARGET_SYMBOLS)
        curr_best = None
        curr_priority = -1
        for node in self.visited_nodes:
            if node.type in priority_map:
                node_priority = priority_map[node.type]
                if node_priority > curr_priority:
                    curr_priority = node_priority
                    curr_best = node
                elif:
                    pass
                else: 
                    pass
        return curr_best
"""

GOLANG_ERROR_SAMPLE_1 = """
func printHello() {
  fmt.Println("hello")
}}

func sayHi() {}
"""

GOLANG_ERROR_SAMPLE_2 = """
func printHello() {
  fmt.Println("hello")
}
}
}

func sayHi() {}
"""

PYTHON_ERROR_SAMPLE_1 = """
s = "text""

def print_hello():
  print("hello")
"""

PYTHON_ERROR_SAMPLE_2 = """
s0 = 'text''
s1='text'
s = 'text'
def print_hello():
    print("hello")
"""


@pytest.mark.parametrize(
    (
        "lang_id",
        "source_code",
        "target_point",
        "expected_context_type",
    ),
    [
        (
            LanguageId.PYTHON,
            PYTHON_SAMPLE,
            (2, 0),
            "list",
        ),
        (
            LanguageId.PYTHON,
            PYTHON_SAMPLE,
            (5, 4),
            "function_definition",
        ),
        (
            LanguageId.PYTHON,
            PYTHON_SAMPLE,
            (22, 16),
            "if_statement",
        ),
        (
            LanguageId.PYTHON,
            PYTHON_SAMPLE,
            (27, 8),
            "for_statement",
        ),
        (
            LanguageId.PYTHON,
            PYTHON_SAMPLE,
            (31, 16),
            "if_statement",
        ),
        (
            LanguageId.PYTHON,
            PYTHON_SAMPLE,
            (34, 16),
            "elif_clause",
        ),
        (
            LanguageId.PYTHON,
            PYTHON_SAMPLE,
            (35, 16),
            "else_clause",
        ),
    ],
)
def test_min_allowed_block_visitor(
    lang_id: LanguageId,
    source_code: str,
    target_point: Point,
    expected_context_type: str,
):
    source_code = source_code.strip("\n")

    parser = CodeParser.from_language_id(
        source_code,
        lang_id,
    )
    visitor = MinAllowedBlockVisitor(target_point)
    tree_dfs(parser.tree, visitor)

    context_node = visitor.block
    assert context_node.type == expected_context_type
    assert context_node.start_point <= target_point <= context_node.end_point


@pytest.mark.parametrize(
    ("lang_id", "source_code", "expected_points"),
    [
        (LanguageId.GO, GOLANG_ERROR_SAMPLE_1, [((0, 0), (2, 2))]),
        (LanguageId.GO, GOLANG_ERROR_SAMPLE_2, [((3, 0), (4, 1))]),
        (
            LanguageId.PYTHON,
            PYTHON_ERROR_SAMPLE_1,
            # consider low-level errors only
            # tree-sitter returns two error nodes for this example:
            # ((0, 0), (3, 16)), ((0, 4), (2, 18))
            [((0, 4), (2, 18))],
        ),
        (
            LanguageId.PYTHON,
            PYTHON_ERROR_SAMPLE_2,
            # consider low-level errors only
            # tree-sitter returns three error nodes for this example:
            # ((0, 3), (3, 17)), ((1, 4), (1, 8)), ((2, 5), (3, 15))
            [((2, 5), (3, 15)), ((1, 4), (1, 8))],
        ),
    ],
)
def test_error_blocks_visitor(
    lang_id: LanguageId,
    source_code: str,
    expected_points: list,
):
    source_code = source_code.strip("\n")

    parser = CodeParser.from_language_id(
        source_code,
        lang_id,
    )
    visitor = ErrorBlocksVisitor()
    tree_dfs(parser.tree, visitor)

    assert all(node.type == "ERROR" for node in visitor.errors)

    actual_points = [(node.start_point, node.end_point) for node in visitor.errors]
    assert actual_points == expected_points
