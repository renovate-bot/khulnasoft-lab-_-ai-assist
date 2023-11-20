import pytest

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.post.ops import fix_end_block_errors

PYTHON_SAMPLE_1 = (
    # prefix
    's = "text"',
    # completion
    '"',
    # suffix
    '\ndef print_hello():\n\tprint("hello")',
)

PYTHON_SAMPLE_2 = (
    # prefix contains the error, counted as one with the error in completion
    # Result: we're not able to fix the completion
    "s = 'text''\ns = 'text'",
    # completion
    "'",
    # suffix
    '\ndef print_hello():\n\tprint("hello")',
)

PYTHON_SAMPLE_3 = (
    # prefix contains the error, counted separately from the error in completion
    # fixing the error in completion, we produce another one
    # Action: we return the initial completion
    "s0 = 'text''\ns1 = 'text'\ns = 'text'",
    # completion
    "'",
    # suffix
    '\ndef print_hello():\n\tprint("hello")',
)

PYTHON_SAMPLE_4 = (
    "def say_hi(s: str",
    # completion
    ")",
    # suffix
    "):",
)

GOLANG_SAMPLE_1 = (
    # prefix
    'func printHello() {\n\tfmt.Println("hello")\n',
    # completion
    "}",
    # suffix
    "}",
)

GOLANG_SAMPLE_2 = (
    # prefix
    'func printHello() {\n\tfmt.Println("hello")\n',
    # completion
    "}}",
    # suffix
    "}",
)

GOLANG_SAMPLE_3 = (
    # prefix
    'func printHello() {\n\tfmt.Println("hello")\n',
    # completion
    "}\n}",
    # suffix
    "}",
)

GOLANG_SAMPLE_4 = (
    # prefix
    'func printHello() {\n\tfmt.Println("hello")\n',
    # completion
    '\tfmt.Println("hello")\n}',
    # suffix
    "}",
)

GOLANG_SAMPLE_5 = (
    # prefix
    'func printHello() {\n\tfmt.Println("hello")\n',
    # completion
    '\tfmt.Println("hello")\n',
    # suffix
    "}",
)


@pytest.mark.parametrize(
    ("code_sample", "lang_id", "expected_completion"),
    [
        (PYTHON_SAMPLE_1, LanguageId.PYTHON, ""),
        (PYTHON_SAMPLE_2, LanguageId.PYTHON, "'"),
        (PYTHON_SAMPLE_3, LanguageId.PYTHON, "'"),
        (PYTHON_SAMPLE_4, LanguageId.PYTHON, ""),
        (GOLANG_SAMPLE_1, LanguageId.GO, ""),
        (GOLANG_SAMPLE_2, LanguageId.GO, ""),
        (GOLANG_SAMPLE_3, LanguageId.GO, ""),
        (GOLANG_SAMPLE_4, LanguageId.GO, '\tfmt.Println("hello")\n'),
        (GOLANG_SAMPLE_5, LanguageId.GO, '\tfmt.Println("hello")\n'),
    ],
)
def test_fix_end_block_errors(
    code_sample: tuple, lang_id: LanguageId, expected_completion: str
):
    prefix, completion, suffix = code_sample
    actual_completion = fix_end_block_errors(
        prefix, completion, suffix, attempts=4, lang_id=lang_id
    )

    assert actual_completion == expected_completion
