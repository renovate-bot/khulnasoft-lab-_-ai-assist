import pytest

from codesuggestions.suggestions.processing.post import ops


@pytest.mark.parametrize(
    ("text", "expected_value"),
    [
        ("first line\nsecond line", "first line\nsecond line"),
        ("```\nfirst line\nsecond line```", "first line\nsecond line"),
        ("```unk\nfirst line\nsecond line```", "first line\nsecond line"),
        ("```unk\nfirst line\nsecond line", "first line\nsecond line"),
        ("\nfirst line\nsecond line```", "\nfirst line\nsecond line"),
        ("```python\none line```", "one line"),
        ("```TypeScript\none line```", "one line"),
        ("```java_script\none line```", "one line"),
        ("```unknown_lang_123\none line```", "one line"),
    ],
)
def test_strip_code_block_markdown(text: str, expected_value: str):
    actual_value = ops.strip_code_block_markdown(text)

    assert actual_value == expected_value
