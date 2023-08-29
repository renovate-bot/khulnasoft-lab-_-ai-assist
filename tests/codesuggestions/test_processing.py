import pytest

from codesuggestions.suggestions.processing import ops
from codesuggestions.suggestions.processing.ops import LanguageId


@pytest.mark.parametrize(
    "file_names,expected_lang_id",
    [
        ([".unknown", "..file"], None),
        (["f.file.c", "f.file.h"], LanguageId.C),
        (
            ["f.cpp", "f.hpp", "f.c++", "f.h++", "f.cc", "f.hh", "f.C", "f.H"],
            LanguageId.CPP,
        ),
        (["f.cs"], LanguageId.CSHARP),
        (["f.go"], LanguageId.GO),
        (["f.java"], LanguageId.JAVA),
        (["f.js", "f.jsx"], LanguageId.JS),
        (["f.php", "f.php3", "f.php4", "f.php5", "f.phps", "f.phpt"], LanguageId.PHP),
        (["f.py"], LanguageId.PYTHON),
        (["f.rb"], LanguageId.RUBY),
        (["f.rs"], LanguageId.RUST),
        (["f.scala"], LanguageId.SCALA),
        (["f.ts", "f.tsx"], LanguageId.TS),
        (["f.kts", "f.kt"], LanguageId.KOTLIN),
    ],
)
def test_resolve_lang_from_filepath(file_names, expected_lang_id):
    for file_name in file_names:
        lang_id = ops.lang_from_filename(file_name)

        assert lang_id == expected_lang_id


@pytest.mark.parametrize(
    "lang_id,prompt,prompt_constructed",
    [
        (None, "model prompt", "model prompt"),
        (LanguageId.C, "model prompt", "<c>model prompt"),
        (LanguageId.CPP, "model prompt", "<cpp>model prompt"),
        (LanguageId.CSHARP, "model prompt", "<csharp>model prompt"),
        (LanguageId.GO, "model prompt", "<go>model prompt"),
        (LanguageId.JAVA, "model prompt", "<java>model prompt"),
        (LanguageId.JS, "model prompt", "<js>model prompt"),
        (LanguageId.PHP, "model prompt", "<php>model prompt"),
        (LanguageId.PYTHON, "model prompt", "<python>model prompt"),
        (LanguageId.RUBY, "model prompt", "<ruby>model prompt"),
        (LanguageId.RUST, "model prompt", "<rust>model prompt"),
        (LanguageId.SCALA, "model prompt", "<scala>model prompt"),
        (LanguageId.TS, "model prompt", "<ts>model prompt"),
        (LanguageId.KOTLIN, "model prompt", "<kotlin>model prompt"),
    ],
)
def test_prepend_lang_id_prompt(lang_id, prompt, prompt_constructed):
    constructed = ops.prepend_lang_id(prompt, lang_id)

    assert constructed == prompt_constructed


@pytest.mark.parametrize(
    "completion,expected_completion",
    [
        ("def hello_world():\n", "def hello_world():"),
        ("def hello_world():\nprint(", "def hello_world():"),
        ("def hello_world():", "def hello_world():"),
        ("\ndef hello_world():", "\ndef hello_world():"),
    ],
)
def test_remove_incomplete_line(completion, expected_completion):
    actual_completion = ops.remove_incomplete_lines(completion, sep="\n")

    assert actual_completion == expected_completion


class TestTrimByMaxLen:
    @pytest.mark.parametrize(
        "prompt,max_length,expected_prompt",
        [
            ("abcdefg", 100, "abcdefg"),
            ("abcdefg", 1, "g"),
        ],
    )
    def test_ok(self, prompt, max_length, expected_prompt):
        actual = self._test_run_processing(prompt, max_length)

        assert actual == expected_prompt

    @pytest.mark.parametrize(
        "prompt,max_length",
        [
            ("abcdefg", 0),
            ("abcdefg", -1),
        ],
    )
    def test_fail(self, prompt, max_length):
        with pytest.raises(ValueError) as _:
            self._test_run_processing(prompt, max_length)

    def _test_run_processing(self, prompt, max_length) -> str:
        return ops.trim_by_max_len(prompt, max_length)


@pytest.mark.parametrize(
    "completion,expected_completion",
    [
        ("random completion```", "random completion"),
        ("random completion```\nanother random text", "random completion"),
        ("```\nanother random text", ""),
        (
            "random completion``another random text",
            "random completion``another random text",
        ),
    ],
)
def test_trim_by_sep(completion, expected_completion):
    actual_completion = ops.trim_by_sep(completion, sep="```")

    assert actual_completion == expected_completion


@pytest.mark.parametrize(
    ("value", "start_index", "expected_point"),
    [
        ("     one line", 0, (0, 5)),
        ("{     one line", 0, (0, 0)),
        ("=     one line", 0, (0, 0)),
        ("     another one line\n", 0, (0, 5)),
        ("one line\n", 0, (0, 0)),
        ("\n\n one line\n", 0, (2, 1)),
        ("\n\n1 line\n2 line", 0, (2, 0)),
        ("\n\n\n", 0, (-1, -1)),
        ("    ", 0, (-1, -1)),
        ("    \n1 line", 5, (1, 0)),
        ("    \n    1 line", 5, (1, 4)),
    ],
)
def test_find_non_whitespace_point(value, start_index, expected_point):
    point = ops.find_non_whitespace_point(value, start_index=start_index)

    assert point == expected_point


@pytest.mark.parametrize(
    ("value", "point", "expected_position"),
    [
        ("one line", (0, 0), 0),
        ("one line", (1, 0), -1),
        ("one line", (0, 9), -1),
        ("one line", (0, 8), 8),
        ("one line", (0, 3), 3),
        ("first line\nsecond line\nthird line", (0, 10), 10),
        ("first line\nsecond line\nthird line", (1, 0), 11),
        ("", (0, 0), -1),
        (None, (0, 0), -1),
        (None, (5, 5), -1),
        ("", (1, 0), -1),
        ("", (0, 1), -1),
        ("one line", (0, 8), 8),
        ("one line\nsecond line", (1, 11), 20),
        ("one line\r\nsecond line", (1, 11), 20),
        ("first\r\nsecond\r\nthird", (2, 4), 17),
        ("first\nsecond\nthird", (2, 4), 17),
    ],
)
def test_find_cursor_position(
    value: str, point: tuple[int, int], expected_position: int
):
    position = ops.find_cursor_position(value, point)

    assert position == expected_position


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
