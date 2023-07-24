import pytest

from codesuggestions.suggestions.processing import ops
from codesuggestions.suggestions.processing.ops import (
    LanguageId,
)


@pytest.mark.parametrize(
    "file_names,expected_lang_id", [
        ([".unknown", "..file"], None),
        (["f.file.c", "f.file.h"], LanguageId.C),
        (["f.cpp", "f.hpp", "f.c++", "f.h++", "f.cc", "f.hh", "f.C", "f.H"], LanguageId.CPP),
        (["f.cs"], LanguageId.CSHARP),
        (["f.go"], LanguageId.GO),
        (["f.java"], LanguageId.JAVA),
        (["f.js"], LanguageId.JS),
        (["f.php", "f.php3", "f.php4", "f.php5", "f.phps", "f.phpt"], LanguageId.PHP),
        (["f.py"], LanguageId.PYTHON),
        (["f.rb"], LanguageId.RUBY),
        (["f.rs"], LanguageId.RUST),
        (["f.scala"], LanguageId.SCALA),
        (["f.ts", "f.tsx"], LanguageId.TS),
        (["f.kts", "f.kt"], LanguageId.KOTLIN),
    ]
)
def test_resolve_lang_from_filepath(file_names, expected_lang_id):
    for file_name in file_names:
        lang_id = ops.lang_from_filename(file_name)

        assert lang_id == expected_lang_id


@pytest.mark.parametrize(
    "lang_id,prompt,prompt_constructed", [
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
    ]
)
def test_prepend_lang_id_prompt(lang_id, prompt, prompt_constructed):
    constructed = ops.prepend_lang_id(prompt, lang_id)

    assert constructed == prompt_constructed


@pytest.mark.parametrize(
    "completion,expected_completion", [
        ("def hello_world():\n", "def hello_world():"),
        ("def hello_world():\nprint(", "def hello_world():"),
        ("def hello_world():", "def hello_world():"),
        ("\ndef hello_world():", "\ndef hello_world():"),
    ]
)
def test_remove_incomplete_line(completion, expected_completion):
    actual_completion = ops.remove_incomplete_lines(completion, sep="\n")

    assert actual_completion == expected_completion


class TestTrimByMaxLen:
    @pytest.mark.parametrize(
        "prompt,max_length,expected_prompt", [
            ("abcdefg", 100, "abcdefg"),
            ("abcdefg", 1, "g"),
        ]
    )
    def test_ok(self, prompt, max_length, expected_prompt):
        actual = self._test_run_processing(prompt, max_length)

        assert actual == expected_prompt

    @pytest.mark.parametrize(
        "prompt,max_length", [
            ("abcdefg", 0),
            ("abcdefg", -1),
        ]
    )
    def test_fail(self, prompt, max_length):
        with pytest.raises(ValueError) as _:
            self._test_run_processing(prompt, max_length)

    def _test_run_processing(self, prompt, max_length) -> str:
        return ops.trim_by_max_len(prompt, max_length)


@pytest.mark.parametrize(
    "completion,expected_completion", [
        ("random completion```", "random completion"),
        ("random completion```\nanother random text", "random completion"),
        ("```\nanother random text", ""),
        ("random completion``another random text", "random completion``another random text"),
    ]
)
def test_trim_by_sep(completion, expected_completion):
    actual_completion = ops.trim_by_sep(completion, sep="```")

    assert actual_completion == expected_completion
