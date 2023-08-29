from pathlib import Path
from typing import Optional

import pytest

from codesuggestions.prompts import PromptTemplate
from codesuggestions.suggestions.processing import CodeContent
from codesuggestions.suggestions.processing.completions import (
    MetadataPromptBuilder,
    _PromptBuilder,
)
from codesuggestions.suggestions.processing.generations import TPL_GENERATION_BASE
from codesuggestions.suggestions.processing.generations import (
    PromptBuilder as PromptBuilderGenerations,
)
from codesuggestions.suggestions.processing.ops import LanguageId


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
        ),
        (
            None,
            "App.vue",
            "<script setup>",
            "\n",
            "This code has a filename of App.vue\n<script setup>",
            "\n",
        ),
    ],
)
def test_completions_prompt_builder(
    lang_id: LanguageId,
    file_name: str,
    prefix: str,
    suffix: str,
    expected_prefix: str,
    expected_suffix: str,
):
    prompt_builder = _PromptBuilder(
        CodeContent(prefix, length_tokens=1),
        CodeContent(suffix, length_tokens=1),
        file_name,
        lang_id=lang_id,
        experiments=None,
    )

    prompt = prompt_builder.build()

    assert prompt.prefix == expected_prefix
    assert prompt.suffix == expected_suffix
    assert isinstance(prompt.metadata, MetadataPromptBuilder)


@pytest.mark.parametrize(
    ("prefix", "file_name", "lang_id"),
    [
        (
            CodeContent(text="# print hello world", length_tokens=1),
            "a/b/c.py",
            LanguageId.PYTHON,
        ),
        (CodeContent(text="# print hello world", length_tokens=1), "a/b/c.unk", None),
        # empty file extension
        (CodeContent(text="# print hello world", length_tokens=1), "a/b/c", None),
    ],
)
def test_generations_prompt_builder(
    prefix: CodeContent, file_name: str, lang_id: Optional[LanguageId]
):
    tpl = PromptTemplate(TPL_GENERATION_BASE)
    builder = PromptBuilderGenerations(prefix, file_name, lang_id=lang_id)
    builder.add_template(tpl)
    prompt = builder.build()

    if lang_id is None:
        file_ext = Path(file_name).suffix.replace(".", "")
        assert file_ext in prompt.prefix
    else:
        lang = lang_id.name.lower()
        assert lang in prompt.prefix

    assert prefix.text in prompt.prefix
    assert isinstance(prompt.metadata, MetadataPromptBuilder)

    assert "{lang}" not in prompt.prefix
    assert "{prefix}" not in prompt.prefix
