import pytest
from unittest.mock import Mock

from codesuggestions.models import TextGenModelOutput
from codesuggestions.suggestions.processing import (
    ops,
    ModelEngineCodegen,
    ModelEnginePalm,
)


def _side_effect_few_shot_tpl(content: str, filename: str, model_output: str):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str):
        assert lang_id.name.lower() in prompt
        assert content in prompt

        return TextGenModelOutput(text=model_output)

    return _fn


def _side_effect_unknown_tpl(content: str, _: str, model_output: str):
    def _fn(prompt: str):
        assert content == prompt

        return TextGenModelOutput(text=model_output)

    return _fn


def _side_effect_lang_prepended(content: str, filename: str, model_output: str):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str):
        assert prompt.startswith(f"<{lang_id.name.lower()}>")
        assert prompt.endswith(content)

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
    _side_effect = model_gen_func(content, file_name, model_output)
    text_gen_base_model.generate = Mock(side_effect=_side_effect)

    engine = ModelEngineCodegen.from_local_templates(
        tpl_codegen_dir,
        text_gen_base_model,
    )

    assert engine.generate_completion(content, file_name) == expected_completion


@pytest.mark.parametrize(
    "content,file_name,model_gen_func,model_output,expected_completion",
    [
        (
            "prompt",
            "f.unk",
            _side_effect_unknown_tpl,
            "random completion",
            "random completion",
        ),
        (
            "prompt",
            "f.unk",
            _side_effect_unknown_tpl,
            "random completion\nnew line",
            "random completion\nnew line",
        ),
        (
            "prompt",
            "f.py",
            _side_effect_lang_prepended,
            "random completion\nnew line",
            "random completion\nnew line",
        ),
    ],
)
def test_model_engine_palm(
    text_gen_base_model,
    content,
    file_name,
    model_gen_func,
    model_output,
    expected_completion,
):
    _side_effect = model_gen_func(content, file_name, model_output)
    text_gen_base_model.generate = Mock(side_effect=_side_effect)

    engine = ModelEnginePalm(text_gen_base_model)

    assert engine.generate_completion(content, file_name) == expected_completion
