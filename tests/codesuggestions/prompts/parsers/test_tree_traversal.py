import pytest

from codesuggestions.prompts.parsers import CodeParser
from codesuggestions.prompts.parsers import tree_bfs
from codesuggestions.suggestions.processing.ops import LanguageId

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
