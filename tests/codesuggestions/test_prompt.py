import pytest

from codesuggestions.suggestions.prompt import LanguageId, LanguageResolver, ModelPromptBuilder


@pytest.mark.parametrize(
    "test_file_names,expected_lang_id", [
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
def test_lang_resolver_from_filepath(test_file_names, expected_lang_id):
    for test_file_name in test_file_names:
        lang_id = LanguageResolver.from_file_name(test_file_name)

        assert lang_id == expected_lang_id


@pytest.mark.parametrize(
    "test_lang_id,prompt,prompt_constructed", [
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
def test_construct_model_prompt_lang(test_lang_id, prompt, prompt_constructed):
    constructed = (
        ModelPromptBuilder(prompt)
        .prepend_lang_id(test_lang_id)
        .prompt
    )

    assert constructed == prompt_constructed
