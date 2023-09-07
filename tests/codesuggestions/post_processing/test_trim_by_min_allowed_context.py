import pytest

from codesuggestions.suggestions.processing.ops import LanguageId, find_cursor_position
from codesuggestions.suggestions.processing.post.ops import trim_by_min_allowed_context

PYTHON_SAMPLE_1 = """
class LineBasedCodeSnippets(BaseCodeSnippetsIterator):
    def __init__(self, content: str, num_lines: int, drop_last: bool = False):
        super().__init__(content)

        self.num_lines = num_lines
        self.drop_last = drop_last

    def _next_snippet(self) -> Iterable[CodeSnippet]:
        content_lines = self.content.splitlines(keepends=False)
        for i in range(0, len(content_lines), self.num_lines):
            snippet_lines = content_lines[i : i + self.num_lines]
            snippet_lines_len = len(snippet_lines)

            if snippet_lines_len < self.num_lines and self.drop_last:
                continue

            yield CodeSnippet(
                text="".join(snippet_lines),
                start_pos=(i, 0),
                end_pos=(i + snippet_lines_len - 1, len(snippet_lines[-1])),
            )
"""

JAVA_SAMPLE_1 = """
public class FactorialCalculator {

    public static void main(String[] args) {
        int number = 5;
        long factorial = calculateFactorial(number);
        System.out.println("Factorial of " + number + " is: " + factorial);
    }

    public static long calculateFactorial(int n) {
        if (n == 0 || n == 1) {
            return 1;
        # do not close this method for testing purposes
"""

RUBY_SAMPLE_1 = """
# frozen_string_literal: true

require 'date'
require_relative 'test_module'

module Gitlab
  class Hello
    def test
      puts "hello"
    end
  end
end
"""


@pytest.mark.parametrize(
    ("code_sample", "point", "lang_id", "expected_range"),
    [
        # Complete by the end of the function `_next_snippet`
        (PYTHON_SAMPLE_1, (8, 0), LanguageId.PYTHON, [(8, 0), (20, 13)]),
        # Complete `continue` only in the `if` block
        (PYTHON_SAMPLE_1, (14, 16), LanguageId.PYTHON, [(14, 16), (14, 24)]),
        # Try to complete the next if block but not the full functions
        # In prod, code-gecko doesn't return large amount of tokens to complete huge functions
        (PYTHON_SAMPLE_1, (12, 0), LanguageId.PYTHON, [(12, 0), (14, 24)]),
        # Return the rest of the file
        (JAVA_SAMPLE_1, (10, 12), LanguageId.JAVA, [(10, 12), (11, 55)]),
        # Complete the `test` function
        (RUBY_SAMPLE_1, (7, 4), LanguageId.RUBY, [(7, 4), (9, 7)]),
    ],
)
def test_trim_by_min_allowed_context(
    code_sample: str, point: tuple[int, int], lang_id: LanguageId, expected_range: list
):
    code_sample = code_sample.strip("\n")
    pos = find_cursor_position(code_sample, point)

    prefix = code_sample[:pos]
    completion = code_sample[pos:]

    expected_start = find_cursor_position(code_sample, expected_range[0])
    expected_end = find_cursor_position(code_sample, expected_range[1])

    actual_string = trim_by_min_allowed_context(prefix, completion, lang_id)
    expected_string = code_sample[expected_start:expected_end]

    assert actual_string == expected_string
