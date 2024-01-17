import pytest
from tree_sitter import Node

from ai_gateway.code_suggestions.processing.ops import LanguageId
from ai_gateway.prompts.parsers import CodeParser, tree_bfs, tree_dfs
from ai_gateway.prompts.parsers.base import BaseVisitor

JAVA_SAMPLE_SOURCE = """
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

// comment 1
@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

/* block comment 1 */
"""

PYTHON_SAMPLE_SOURCE = """import os"""


@pytest.mark.parametrize(
    ("lang_id", "source_code", "max_depth", "expected_node_count"),
    [
        (LanguageId.JAVA, "", 0, 0),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 0, 5),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 1, 15),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 2, 26),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 3, 39),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 4, 53),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 5, 60),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 6, 66),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 7, 73),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 8, 76),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 9, 76),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 10, 76),
        (LanguageId.PYTHON, PYTHON_SAMPLE_SOURCE, 0, 1),
        (LanguageId.PYTHON, PYTHON_SAMPLE_SOURCE, 1, 3),
        (LanguageId.PYTHON, PYTHON_SAMPLE_SOURCE, 2, 4),
        (LanguageId.PYTHON, PYTHON_SAMPLE_SOURCE, 3, 4),
    ],
)
@pytest.mark.asyncio
async def test_level_order_traversal(
    lang_id: LanguageId, source_code: str, max_depth: int, expected_node_count: int
):
    root_node = (await CodeParser.from_language_id(source_code, lang_id)).tree.root_node

    visited_nodes = []

    def collect_nodes(_depth, node):
        visited_nodes.append(node)

    tree_bfs(root_node, collect_nodes, max_depth=max_depth)
    assert len(visited_nodes) == expected_node_count


class StubSimpleVisitor(BaseVisitor):
    def __init__(self):
        super().__init__()
        self.visited_nodes = []

    def _visit_node(self, node: Node):
        self.visited_nodes.append(node)

    def visit(self, node: Node):
        self._visit_node(node)


class StubLimitedDepthVisitor(BaseVisitor):
    def __init__(self, stop_limit=-1):
        self.visited_nodes = []
        self.stop_limit = stop_limit

    def _visit_node(self, node: Node):
        self.visited_nodes.append(node)

    @property
    def stop_tree_traversal(self) -> bool:
        return 0 < self.stop_limit <= len(self.visited_nodes)

    def visit(self, node: Node):
        self._visit_node(node)


class StubLimitedNodeTraversalVisitor(BaseVisitor):
    def __init__(self, symbol: str):
        self.visited_nodes = []
        self.symbol = symbol
        self._stop_node_traversal = False

    def _visit_node(self, node: Node):
        self.visited_nodes.append(node)
        self._stop_node_traversal = True if node.type == self.symbol else False

    @property
    def stop_node_traversal(self) -> bool:
        return self._stop_node_traversal

    def visit(self, node: Node):
        self._visit_node(node)


@pytest.mark.parametrize(
    ("lang_id", "source_code", "visitor", "expected_node_count"),
    [
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, StubLimitedDepthVisitor(-1), 77),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, StubLimitedDepthVisitor(50), 50),
        (
            LanguageId.JAVA,
            JAVA_SAMPLE_SOURCE,
            StubLimitedNodeTraversalVisitor("class_declaration"),
            33,
        ),
        (
            LanguageId.JAVA,
            JAVA_SAMPLE_SOURCE,
            StubLimitedNodeTraversalVisitor("import_declaration"),
            50,
        ),
    ],
)
@pytest.mark.asyncio
async def test_preorder_traversal(
    lang_id: LanguageId,
    source_code: str,
    visitor: StubLimitedDepthVisitor,
    expected_node_count: int,
):
    tree = (await CodeParser.from_language_id(source_code, lang_id)).tree

    tree_dfs(tree, visitor)

    assert len(visitor.visited_nodes) == expected_node_count


@pytest.mark.parametrize(
    ("lang_id", "source_code", "visitor", "max_visit_count", "expected_node_count"),
    [
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, StubSimpleVisitor(), 2, 2),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, StubSimpleVisitor(), 80, 77),
    ],
)
@pytest.mark.asyncio
async def test_tree_dfs_max_visit_count(
    lang_id: LanguageId,
    source_code: str,
    visitor: StubSimpleVisitor,
    max_visit_count: int,
    expected_node_count: int,
):
    tree = (await CodeParser.from_language_id(source_code, lang_id)).tree

    tree_dfs(tree, visitor, max_visit_count=max_visit_count)

    assert len(visitor.visited_nodes) == expected_node_count
