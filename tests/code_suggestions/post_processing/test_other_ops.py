import pytest

from ai_gateway.code_suggestions.processing.post import ops
from ai_gateway.code_suggestions.processing.post.ops import (
    remove_comment_only_completion,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId


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


@pytest.mark.parametrize(
    ("code_context", "completion", "expected_value"),
    [
        ("", "", ""),
        ("code context", "", ""),
        ("code context", "\ncompletion", "\ncompletion"),
        ("code context", "completion", "\ncompletion"),
        ("code context\n", "completion", "completion"),
    ],
)
def test_prepend_new_line(code_context: str, completion: str, expected_value: str):
    actual_value = ops.prepend_new_line(code_context, completion)

    assert actual_value == expected_value


@pytest.mark.parametrize(
    ("completion", "lang_id", "expected"),
    [
        (
            'if __name__=="__main__":\n\tprint(f"Hello world!")',
            LanguageId.PYTHON,
            'if __name__=="__main__":\n\tprint(f"Hello world!")',
        ),
        (
            "# This function prints 'hello'\ndef hello():\n\tprint('hello')\n",
            LanguageId.PYTHON,
            "# This function prints 'hello'\ndef hello():\n\tprint('hello')\n",
        ),
        (
            "# This is just a comment\n# followed by another comment",
            LanguageId.PYTHON,
            "",
        ),
        (
            "this is a line being completed')\n\n",
            LanguageId.PYTHON,
            "this is a line being completed')\n\n",
        ),
    ],
)
@pytest.mark.asyncio
async def test_remove_comment_only_completion(
    completion: str, lang_id: LanguageId, expected: str
):
    actual = await remove_comment_only_completion(completion, lang_id)

    assert actual == expected
