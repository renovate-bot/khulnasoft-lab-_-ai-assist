from typing import Any

import pytest
from unittest.mock import Mock, AsyncMock, PropertyMock
from contextlib import contextmanager
from transformers import AutoTokenizer

from codesuggestions.models import (
    TextGenModelOutput,
    PalmCodeGenBaseModel,
    VertexModelInternalError,
    VertexModelInvalidArgument,
)
from codesuggestions.suggestions.processing import (
    ops,
    ModelEngineCodegen,
    ModelEnginePalm, MetadataModel, MetadataPromptBuilder, MetadataCodeContent, MetadataImports,
)

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-16B")


class MockInstrumentor:
    @contextmanager
    def watch(self, prompt: str, **kwargs: Any):
        yield Mock()


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


def _side_effect_unknown_tpl_palm(prefix: str, _suffix: str, filename: str, model_output: str):
    def _fn(prompt: str, _suffix: str):
        assert filename in prompt

        return TextGenModelOutput(text=model_output)

    return _fn


def _side_effect_lang_prepended(content: str, _suffix: str, filename: str, model_output: str):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str, _suffix: str):
        assert prompt.startswith(f"<{lang_id.name.lower()}>")
        assert prompt.endswith(content)

        return TextGenModelOutput(text=model_output)

    return _fn


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


def _side_effect_with_internal_exception(content: str, suffix: str, filename: str, model_output: str):
    def _fn(prompt: str, suffix: str):
        raise VertexModelInternalError("internal error")

    return _fn


def _side_effect_with_invalid_arg_exception(content: str, suffix: str, filename: str, model_output: str):
    def _fn(prompt: str, suffix: str):
        raise VertexModelInvalidArgument("invalid argument")

    return _fn


def token_length(s: str):
    return len(tokenizer(s)["input_ids"])


@pytest.mark.asyncio
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
async def test_model_engine_codegen(
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

    completion = await engine.generate_completion(content, "", file_name)

    assert completion.text == expected_completion
    assert completion.model is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix,suffix,file_name,model_gen_func,model_output,"
    "language,prompt_builder_metadata,expected_completion",
    [
        (
            "prompt",
            "",
            "f.unk",
            _side_effect_unknown_tpl_palm,
            "random completion",
            None,
            MetadataPromptBuilder(
                prefix=MetadataCodeContent(length=6, length_tokens=2),
                suffix=MetadataCodeContent(length=0, length_tokens=0),
                imports=MetadataImports(pre=MetadataCodeContent(length=0, length_tokens=0),
                                        post=MetadataCodeContent(length=0, length_tokens=0)),
            ),
            "random completion",
        ),
        (
            "prompt",
            "",
            "f.unk",
            _side_effect_unknown_tpl_palm,
            "random completion\nnew line",
            None,
            MetadataPromptBuilder(
                prefix=MetadataCodeContent(length=6, length_tokens=2),
                suffix=MetadataCodeContent(length=0, length_tokens=0),
                imports=MetadataImports(pre=MetadataCodeContent(length=0, length_tokens=0),
                                        post=MetadataCodeContent(length=0, length_tokens=0)),
            ),
            "random completion\nnew line",
        ),
        (
            "prompt " * 2048,
            "abc " * 4096,
            "f.py",
            _side_effect_with_suffix,
            "random completion\nnew line",
            ops.LanguageId.PYTHON,
            MetadataPromptBuilder(
                prefix=MetadataCodeContent(length=3494, length_tokens=500),
                suffix=MetadataCodeContent(length=1002, length_tokens=500),
                imports=MetadataImports(pre=MetadataCodeContent(length=0, length_tokens=0),
                                        post=MetadataCodeContent(length=0, length_tokens=0))
            ),
            "random completion\nnew line",
        ),
        (
            "import os\nimport pytest\n" + "prompt" * 2048,
            "",
            "f.py",
            _side_effect_with_imports,
            "random completion\nnew line",
            ops.LanguageId.PYTHON,
            MetadataPromptBuilder(
                prefix=MetadataCodeContent(length=2984, length_tokens=995),
                suffix=MetadataCodeContent(length=0, length_tokens=0),
                imports=MetadataImports(pre=MetadataCodeContent(length=22, length_tokens=5),
                                        post=MetadataCodeContent(length=22, length_tokens=5))
            ),
            "random completion\nnew line",
        ),
        (
            "random_prefix",
            "random_suffix",
            "f.unk",
            _side_effect_with_internal_exception,
            "unreturned completion due to an exception",
            None,
            None,
            "",
        ),
(
            "random_prefix",
            "random_suffix",
            "f.unk",
            _side_effect_with_invalid_arg_exception,
            "unreturned completion due to an exception",
            None,
            None,
            "",
        ),
    ],
)
async def test_model_engine_palm(
    text_gen_base_model,
    prefix,
    suffix,
    file_name,
    model_gen_func,
    model_output,
    language,
    prompt_builder_metadata,
    expected_completion,
):
    model_name = "palm-model"
    model_engine = "vertex-ai"
    _side_effect = model_gen_func(prefix, suffix, file_name, model_output)

    text_gen_base_model.generate = AsyncMock(side_effect=_side_effect)
    type(text_gen_base_model).model_name = PropertyMock(return_value=model_name)
    type(text_gen_base_model).model_engine = PropertyMock(return_value=model_engine)

    engine = ModelEnginePalm(text_gen_base_model, tokenizer)
    engine.instrumentator = MockInstrumentor()
    completion = await engine.generate_completion(prefix, suffix, file_name)

    assert completion.text == expected_completion
    assert completion.model == MetadataModel(name=model_name, engine=model_engine)
    assert completion.lang_id == language
    assert completion.metadata == prompt_builder_metadata
