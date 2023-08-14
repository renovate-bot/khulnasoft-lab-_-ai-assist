import pytest

from codesuggestions.prompts.parsers import CodeParser, MinAllowedBlockVisitor, Point
from codesuggestions.prompts.parsers.treetraversal import tree_dfs
from codesuggestions.suggestions.processing.ops import LanguageId

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
            "class_definition",
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
