import pytest

from codesuggestions.suggestions.processing.post_processor import PostProcessor


class TestPostProcessor:
    @pytest.mark.parametrize(
        "completion,expected_output",
        [
            ("        ", ""),
            ("\n    \t", ""),
            ("\n hello \t world", "\n hello \t world"),
        ],
    )
    def test_process(self, completion, expected_output):
        processor = PostProcessor()
        actual = processor.process(completion)

        assert actual == expected_output
