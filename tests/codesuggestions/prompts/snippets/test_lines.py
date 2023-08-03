import pytest

from codesuggestions.prompts.snippets import CodeSnippet, LineBasedCodeSnippets

PYTHON_SOURCE_SAMPLE = """
import os
import time

# 1 comment
# 2 comment
# 3 comment
# 4 comment
# 5 comment
# 6 comment

import random
import pandas as pd

def sum(a, b):
    import numpy as np
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    def __init__(self):
        self.result = 0

    def calculate_sum(self, a, b):
        self.result = sum(a, b)
"""

PYTHON_SNIPPETS_LINES_5 = [
    CodeSnippet(
        text="import os\nimport time\n\n# 1 comment\n# 2 comment",
        start_pos=(0, 0),
        end_pos=(4, 11),
    ),
    CodeSnippet(
        text="# 3 comment\n# 4 comment\n# 5 comment\n# 6 comment\n",
        start_pos=(5, 0),
        end_pos=(9, 0),
    ),
    CodeSnippet(
        text="import random\nimport pandas as pd\n\ndef sum(a, b):\n    import numpy as np",
        start_pos=(10, 0),
        end_pos=(14, 22),
    ),
    CodeSnippet(
        text="    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        start_pos=(15, 0),
        end_pos=(19, 0),
    ),
    CodeSnippet(
        text="class Calculator:\n    def __init__(self):\n        self.result = 0\n\n    def calculate_sum(self, a, b):",
        start_pos=(20, 0),
        end_pos=(24, 34),
    ),
    CodeSnippet(
        text="        self.result = sum(a, b)", start_pos=(25, 0), end_pos=(25, 31)
    ),
]


@pytest.mark.parametrize(
    ("content", "num_lines", "drop_last", "expected_snippets"),
    [
        (PYTHON_SOURCE_SAMPLE.strip("\n"), 5, False, PYTHON_SNIPPETS_LINES_5),
        (PYTHON_SOURCE_SAMPLE.strip("\n"), 5, True, PYTHON_SNIPPETS_LINES_5[:-1]),
        ("", 5, False, []),
        (
            "random content",
            100,
            False,
            [CodeSnippet(text="random content", start_pos=(0, 0), end_pos=(0, 14))],
        ),
        ("random content", 100, True, []),
    ],
)
def test_code_snippets_line_based(
    content: str, num_lines: int, drop_last: bool, expected_snippets: list
):
    # collect all snippets for tests at once
    code_snippets = list(LineBasedCodeSnippets(content, num_lines, drop_last))

    assert code_snippets == expected_snippets
