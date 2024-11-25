import pytest

from ai_gateway.code_suggestions.processing import ops
from ai_gateway.code_suggestions.processing.ops import LanguageId


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
    "editor_langs,expected_lang_id",
    [
        (["unknown", "file"], None),
        (["c"], LanguageId.C),
        (["cpp"], LanguageId.CPP),
        (["csharp"], LanguageId.CSHARP),
        (["go"], LanguageId.GO),
        (["java"], LanguageId.JAVA),
        (["javascript", "javascriptreact"], LanguageId.JS),
        (["php"], LanguageId.PHP),
        (["python"], LanguageId.PYTHON),
        (["ruby"], LanguageId.RUBY),
        (["rust"], LanguageId.RUST),
        (["scala"], LanguageId.SCALA),
        (["typescript", "typescriptreact"], LanguageId.TS),
        (["kotlin"], LanguageId.KOTLIN),
    ],
)
def test_resolve_lang_from_editor_name(editor_langs, expected_lang_id):
    for name in editor_langs:
        lang_id = ops.lang_from_editor_lang(name)

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
    ("value", "start_index", "expected_position"),
    [
        ("one line\nanother line", 0, 9),
        ("one line\nanother line", 8, 9),
        ("one line\nanother line", 5, 9),
        ("one line\nanother line", 9, 9),
        ("one line\nanother line\n", 0, 9),
        ("one line\nanother line\n", 0, 9),
        ("one line\n\tanother line\n", 0, 9),
        ("one line\n\tanother line\n", 10, 10),
        ("one line\n\t\tanother line\n", 11, 11),
        ("one line\nanother line", 10, -1),
        ("one line", 0, -1),
    ],
)
def test_find_newline_position(value: str, start_index: int, expected_position: int):
    actual = ops.find_newline_position(value, start_index=start_index)

    assert actual == expected_position


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ([], [], []),
        ([], ["abc"], []),
        (["abc"], [], []),
        (["abc"], ["def"], []),
        (["abc", "def"], ["abc", "def"], [(0, 1)]),
        (["abc", "abc", "a", "abc", "def"], ["abc", "def", "a"], [(0, 1), (2,)]),
        (
            ["b", "abc", "def", "a", "abc", "def"],
            ["abc", "def", "c", "abc", "def", "k"],
            [(0, 1), (3, 4)],
        ),
        (["abc"], ["abc"], [(0,)]),
    ],
)
def test_find_common_lines(source: list, target: list, expected: list):
    actual = ops.find_common_lines(source, target)

    assert actual == expected


@pytest.mark.parametrize(
    "completion,expected_output",
    [
        ("        ", ""),
        ("\n    \t", ""),
        ("\n hello \t world", "\n hello \t world"),
    ],
)
@pytest.mark.asyncio
async def test_strip_whitespaces(completion, expected_output):
    actual = await ops.strip_whitespaces(completion)

    assert actual == expected_output


@pytest.mark.parametrize(
    "completion,max_trim,expected_completion",
    [
        ("def hello_world():\n\tfoo=1", 0.5, "def hello_world():\n\tfoo=1"),
        (
            "Path)\n\n\treturn dirPath\n}\n\nfunc (p *PsWriter) RmDir(path string) {",
            0.8,
            "func (p *PsWriter) RmDir(path string) {",
        ),
        (
            "Path)\n\n\treturn dirPath\n}\n\nfunc (p *PsWriter) RmDir(path string) {",
            0.2,
            "Path)\n\n\treturn dirPath\n}\n\nfunc (p *PsWriter) RmDir(path string) {",
        ),
        (
            "Path)\n\n\treturn dirPath\n}\n\n// some comment\n\nfunc (p *PsWriter) RmDir(path string) {",
            0.8,
            "// some comment\n\nfunc (p *PsWriter) RmDir(path string) {",
        ),
        (
            "Path)\n\n\treturn dirPath\n}\n\n/* some\nlong\ncomment */\n\nfunc (p *PsWriter) RmDir(path string) {",
            0.8,
            "/* some\nlong\ncomment */\n\nfunc (p *PsWriter) RmDir(path string) {",
        ),
        (
            "Path)\n\n\treturn dirPath\n}\n\n// some comment\n\nfunc (p *PsWriter) RmDir(path string) {",
            0.8,
            "// some comment\n\nfunc (p *PsWriter) RmDir(path string) {",
        ),
        (
            "Path)\n\n\treturn dirPath\n}\n\n# some comment\n\nfunc (p *PsWriter) RmDir(path string) {",
            0.8,
            "# some comment\n\nfunc (p *PsWriter) RmDir(path string) {",
        ),
        (
            "Path)\n\n\treturn dirPath\n}\n\n# multiple\n\n// comments\n\n/* foo\nbar\n*/\n\nfunc (p *PsWriter) RmDir(path string) {",
            0.8,
            "# multiple\n\n// comments\n\n/* foo\nbar\n*/\n\nfunc (p *PsWriter) RmDir(path string) {",
        ),
    ],
)
def test_remove_incomplete_block(
    completion: str, max_trim: float, expected_completion: str
):
    actual_completion = ops.remove_incomplete_block(
        s=completion, max_trim_percent=max_trim
    )

    assert actual_completion == expected_completion


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("", "", True),
        ("abc", "abc", True),
        ("abc", "abcd", False),
        ("abc", "Abc", False),
    ],
)
def test_compare_exact(source: str, target: str, expected: bool):
    actual = ops.compare_exact(a=source, b=target)

    assert actual == expected
