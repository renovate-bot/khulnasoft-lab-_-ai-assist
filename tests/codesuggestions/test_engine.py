import pytest
from unittest.mock import Mock

from codesuggestions.models import TextGenModelOutput, PalmCodeGenBaseModel
from codesuggestions.suggestions.processing import (
    ops,
    ModelEngineCodegen,
    ModelEnginePalm,
)

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")


def _side_effect_few_shot_tpl(content: str, _suffix: str, filename: str, model_output: str):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str, _suffix: str):
        assert lang_id.name.lower() in prompt
        assert content in prompt

        return TextGenModelOutput(text=model_output)

    return _fn


def _side_effect_unknown_tpl(content: str, _suffix: str, _: str, model_output: str):
    def _fn(prompt: str, _suffix: str):
        assert content == prompt

        return TextGenModelOutput(text=model_output)

    return _fn


def _side_effect_lang_prepended(content: str, _suffix: str, filename: str, model_output: str):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str, _suffix: str):
        assert prompt.startswith(f"<{lang_id.name.lower()}>")
        assert prompt.endswith(content)

        return TextGenModelOutput(text=model_output)

    return _fn


def token_length(s: str):
    tokens = tokenizer(s, return_length=True)
    return tokens['length']


def _side_effect_with_suffix(content: str, suffix: str, filename: str, model_output: str):
    original_suffix = suffix

    def _fn(prompt: str, suffix: str):
        assert original_suffix.startswith(suffix)
        assert token_length(prompt) + token_length(suffix) <= PalmCodeGenBaseModel.MAX_MODEL_LEN

        return TextGenModelOutput(text=model_output)

    return _fn


def _side_effect_with_imports(content: str, suffix: str, filename: str, model_output: str):
    def _fn(prompt: str, suffix: str):
        assert content.startswith("import os\nimport pytest")

        return TextGenModelOutput(text=model_output)

    return _fn


@pytest.mark.parametrize(
    "content,file_name,model_gen_func,model_output,expected_completion",
    [
        (
            "random prompt",
            "filename.py",
            _side_effect_few_shot_tpl,
            "random completion",
            "random completion",
        ),
        (
            "random prompt",
            "filename.py",
            _side_effect_few_shot_tpl,
            "random completion\nnew line",
            "random completion",
        ),
        (
            "random prompt",
            "filename.py",
            _side_effect_few_shot_tpl,
            "random completion```",
            "random completion",
        ),
        (
            "random prompt",
            "filename.py",
            _side_effect_few_shot_tpl,
            "random completion```\nnew line",
            "random completion",
        ),
        ("random prompt", "filename.py", _side_effect_few_shot_tpl, "", ""),
        (
            "random prompt",
            "filename.unk",
            _side_effect_unknown_tpl,
            "random completion",
            "random completion",
        ),
        (
            "random prompt",
            "filename.unk",
            _side_effect_unknown_tpl,
            "random completion\nnew line",
            "random completion",
        ),
    ],
)
def test_model_engine_codegen(
    text_gen_base_model,
    tpl_codegen_dir,
    content,
    file_name,
    model_gen_func,
    model_output,
    expected_completion,
):
    _side_effect = model_gen_func(content, '', file_name, model_output)
    text_gen_base_model.generate = Mock(side_effect=_side_effect)

    engine = ModelEngineCodegen.from_local_templates(
        tpl_codegen_dir,
        text_gen_base_model,
    )

    assert engine.generate_completion(content, "", file_name) == expected_completion


@pytest.mark.parametrize(
    "content,suffix,file_name,model_gen_func,model_output,expected_completion",
    [
        (
            "prompt",
            "",
            "f.unk",
            _side_effect_unknown_tpl,
            "random completion",
            "random completion",
        ),
        (
            "prompt",
            "",
            "f.unk",
            _side_effect_unknown_tpl,
            "random completion\nnew line",
            "random completion\nnew line",
        ),
        # TODO: Disable this test as we temporarily do not prepend with lang_id
        # (
        #     "prompt",
        #     "",
        #     "f.py",
        #     _side_effect_lang_prepended,
        #     "random completion\nnew line",
        #     "random completion\nnew line",
        # ),
        (
            "prompt " * 2048,
            "abc " * 4096,
            "f.py",
            _side_effect_with_suffix,
            "random completion\nnew line",
            "random completion\nnew line",
        ),
        (
            "import os\nimport pytest\n" + "prompt" * 2048,
            "",
            "f.py",
            _side_effect_with_imports,
            "random completion\nnew line",
            "random completion\nnew line",
        ),
    ],
)
def test_model_engine_palm(
    text_gen_base_model,
    content,
    suffix,
    file_name,
    model_gen_func,
    model_output,
    expected_completion,
):
    _side_effect = model_gen_func(content, suffix, file_name, model_output)
    text_gen_base_model.generate = Mock(side_effect=_side_effect)

    engine = ModelEnginePalm(text_gen_base_model, tokenizer)

    assert engine.generate_completion(content, suffix, file_name) == expected_completion
