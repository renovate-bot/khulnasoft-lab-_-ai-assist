import pytest

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.post.ops import (
    fix_end_block_errors,
    fix_end_block_errors_with_comparison,
)

PYTHON_SAMPLE_1 = (
    # prefix
    's = "text',
    # completion
    '"',
    # suffix
    '"\ndef print_hello():\n\tprint("hello")',
)

PYTHON_SAMPLE_2 = (
    # prefix, contains the error
    "s = 'text''\ns = 'text",
    # completion
    "'",
    # suffix
    "'\ndef print_hello():\n\tprint('hello')",
)

PYTHON_SAMPLE_3 = (
    # prefix
    "def say_hi(s: str",
    # completion
    "):",
    # suffix
    "):",
)

PYTHON_SAMPLE_4 = (
    # prefix
    "def say_hi(s: str",
    # completion
    "v: int):",
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
    '\tfmt.Println("hello")\n}',
    # suffix
    "}",
)

GOLANG_SAMPLE_4 = (
    # prefix
    'func printHello() {\n\tfmt.Println("hello")\n',
    # completion
    '\tfmt.Println("hello")\n',
    # suffix
    "}\n\nfunc printHello() {\n\tfmt.Println('hello')\n",
)

RUBY_SAMPLE_1 = (
    # prefix
    "def print_hello\n\t",
    # completion
    "puts 'hello'\nend",
    # suffix
    "\nend",
)

RUBY_SAMPLE_2 = (
    # prefix
    "def print_hello(\n\t",
    # completion has extra end block
    "puts 'hello'\nend",
    # suffix
    "\nend",
)

RUBY_SAMPLE_3 = (
    # prefix has error
    "def print_hello(\n\t",
    # completion
    "puts 'hello'",
    # suffix
    "\nend",
)


@pytest.mark.parametrize(
    ("code_sample", "lang_id", "expected_completion"),
    [
        # completion has extra end block; the current approach can fix the error
        (PYTHON_SAMPLE_1, LanguageId.PYTHON, ""),
        # prefix contains errors; completion extra end block cannot be trimmed
        (PYTHON_SAMPLE_2, LanguageId.PYTHON, "'"),
        # completion has unnecessary end blocks, we're able to trim it
        (PYTHON_SAMPLE_3, LanguageId.PYTHON, ""),
        # completion causes errors unrelated to extra end blocks, it is not fixed
        (PYTHON_SAMPLE_4, LanguageId.PYTHON, "v: int"),
        # completion has extra end block; the current approach can fix the error
        (GOLANG_SAMPLE_1, LanguageId.GO, ""),
        # completion contains the error; the current approach is not able to fix the error in one attempt
        (GOLANG_SAMPLE_2, LanguageId.GO, "}}"),
        # completion has extra end block; the current approach can fix the error
        (GOLANG_SAMPLE_3, LanguageId.GO, '\tfmt.Println("hello")\n'),
        # completion is correct; no post-processing done
        (GOLANG_SAMPLE_4, LanguageId.GO, '\tfmt.Println("hello")\n'),
        # completion has extra end block; the current approach can trim the end block
        (RUBY_SAMPLE_1, LanguageId.RUBY, "puts 'hello'\n"),
        # the prefix has an error; completion extra end block cannot be trimmed
        (RUBY_SAMPLE_2, LanguageId.RUBY, "puts 'hello'\nend"),
        # completion is correct; no post-processing done
        (RUBY_SAMPLE_3, LanguageId.RUBY, "puts 'hello'"),
    ],
)
@pytest.mark.asyncio
async def test_fix_end_block_errors(
    code_sample: tuple, lang_id: LanguageId, expected_completion: str
):
    prefix, completion, suffix = code_sample
    actual_completion = await fix_end_block_errors(
        prefix, completion, suffix, lang_id=lang_id
    )

    assert actual_completion == expected_completion


@pytest.mark.parametrize(
    ("code_sample", "lang_id", "expected_completion"),
    [
        # completion has extra end block; the current approach can fix the error
        (PYTHON_SAMPLE_1, LanguageId.PYTHON, ""),
        # prefix contains errors;
        # the current approach removes extra content since the error is unrelated to the completion
        (PYTHON_SAMPLE_2, LanguageId.PYTHON, ""),
        # completion has unnecessary end blocks, we're able to trim it
        (PYTHON_SAMPLE_3, LanguageId.PYTHON, ""),
        # completion causes errors unrelated to extra end blocks, it is not fixed
        (PYTHON_SAMPLE_4, LanguageId.PYTHON, "v: int"),
        # completion has extra end block; the current approach can fix the error
        (GOLANG_SAMPLE_1, LanguageId.GO, ""),
        # the completion contains 2 extra end blocks; the current approach is not able to fix the error
        (GOLANG_SAMPLE_2, LanguageId.GO, "}}"),
        # completion has extra end block; the current approach can fix the error
        (GOLANG_SAMPLE_3, LanguageId.GO, '\tfmt.Println("hello")'),
        # completion is correct
        (GOLANG_SAMPLE_4, LanguageId.GO, '\tfmt.Println("hello")\n'),
        # completion has extra end block; the current approach can trim the end block and trailing spaces
        (RUBY_SAMPLE_1, LanguageId.RUBY, "puts 'hello'"),
        # the prefix has an error; the current approach can trim the end block and trailing spaces
        (RUBY_SAMPLE_2, LanguageId.RUBY, "puts 'hello'"),
        # completion is correct; no post-processing done
        (RUBY_SAMPLE_3, LanguageId.RUBY, "puts 'hello'"),
    ],
)
@pytest.mark.asyncio
async def test_fix_end_block_errors_with_comparison(
    code_sample: tuple, lang_id: LanguageId, expected_completion: str
):
    prefix, completion, suffix = code_sample
    actual_completion = await fix_end_block_errors_with_comparison(
        prefix, completion, suffix, lang_id=lang_id
    )

    assert actual_completion == expected_completion
