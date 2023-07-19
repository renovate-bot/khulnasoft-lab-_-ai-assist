import pytest

from codesuggestions.suggestions.processing.base import LanguageId
from codesuggestions.suggestions.processing.engine import _PromptBuilder


@pytest.mark.parametrize(
    "lang_id,file_name,prefix,suffix,expected_prefix,expected_suffix",
    [
        (
            LanguageId.C,
            "test.c",
            "int main()",
            "}",
            "/* This code has a filename of test.c and is written in C. */\nint main()",
            "}",
        ),
        (
            LanguageId.PYTHON,
            "test.py",
            "def sort(arr):",
            "\n",
            "# This code has a filename of test.py and is written in Python.\ndef sort(arr):",
            "\n",
        )
    ]
)
def test_prompt_builder(
    lang_id: LanguageId,
    file_name: str,
    prefix: str,
    suffix: str,
    expected_prefix: str,
    expected_suffix: str
):
    prompt_builder = _PromptBuilder(lang_id, file_name, prefix, suffix)
    prefix, suffix = prompt_builder.build()

    assert prefix == expected_prefix
    assert suffix == expected_suffix
