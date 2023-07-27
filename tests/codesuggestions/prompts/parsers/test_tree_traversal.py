import pytest

from codesuggestions.prompts.parsers import CodeParser
from codesuggestions.prompts.parsers import tree_bfs, tree_dfs
from codesuggestions.suggestions.processing.ops import LanguageId
from codesuggestions.prompts.parsers.base import BaseVisitor

from tree_sitter import Node

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
    ]
)
def test_level_order_traversal(lang_id: LanguageId, source_code: str, max_depth: int, expected_node_count: int):
    root_node = CodeParser.from_language_id(source_code, lang_id).tree.root_node

    visited_nodes = []

    def collect_nodes(_depth, node):
        visited_nodes.append(node)

    tree_bfs(root_node, collect_nodes, max_depth=max_depth)
    assert len(visited_nodes) == expected_node_count


class StubVisitor(BaseVisitor):
    def __init__(self, stop_limit=-1):
        self.visited_nodes = []
        self.stop_limit = stop_limit

    def _visit_node(self, node: Node):
        pass

    @property
    def stop_earlier(self) -> bool:
        return 0 < self.stop_limit <= len(self.visited_nodes)

    def visit(self, node: Node):
        self.visited_nodes.append(node)


@pytest.mark.parametrize(
    ("lang_id", "source_code", "stop_limit", "expected_node_count"),
    [
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, -1, 77),
        (LanguageId.JAVA, JAVA_SAMPLE_SOURCE, 50, 50),
    ]
)
def test_preorder_traversal(lang_id: LanguageId, source_code: str, stop_limit: int, expected_node_count: int):
    tree = CodeParser.from_language_id(source_code, lang_id).tree
    visitor = StubVisitor(stop_limit)

    tree_dfs(tree, visitor)

    assert len(visitor.visited_nodes) == expected_node_count
