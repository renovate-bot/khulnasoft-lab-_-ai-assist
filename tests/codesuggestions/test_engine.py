from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, PropertyMock

import pytest
from transformers import AutoTokenizer

from codesuggestions.models import (
    PalmCodeGenBaseModel,
    TextGenModelOutput,
    VertexModelInternalError,
    VertexModelInvalidArgument,
)
from codesuggestions.suggestions.processing import (
    MetadataCodeContent,
    MetadataExtraInfo,
    MetadataModel,
    MetadataPromptBuilder,
    ModelEngineCodegen,
    ModelEnginePalm,
    ops,
)

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-16B")


class MockInstrumentor:
    def __init__(self):
        self.watcher = Mock()
        self.watcher.register_prompt_symbols = Mock()
        self.watcher.register_model_output_length = Mock()

    @contextmanager
    def watch(self, prompt: str, **kwargs: Any):
        yield self.watcher


def _side_effect_few_shot_tpl(
    content: str, _suffix: str, filename: str, model_output: str
):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str, _suffix: str):
        assert lang_id.name.lower() in prompt
        assert content in prompt

        return TextGenModelOutput(text=model_output, score=-1)

    return _fn


def _side_effect_unknown_tpl(content: str, _suffix: str, _: str, model_output: str):
    def _fn(prompt: str, _suffix: str):
        assert content == prompt

        return TextGenModelOutput(text=model_output, score=-1)

    return _fn


def _side_effect_unknown_tpl_palm(
    prefix: str, _suffix: str, filename: str, model_output: str
):
    def _fn(prompt: str, _suffix: str):
        assert filename in prompt

        return TextGenModelOutput(text=model_output, score=-1)

    return _fn


def _side_effect_lang_prepended(
    content: str, _suffix: str, filename: str, model_output: str
):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str, _suffix: str):
        assert prompt.startswith(f"<{lang_id.name.lower()}>")
        assert prompt.endswith(content)

        return TextGenModelOutput(text=model_output, score=-1)

    return _fn


def _side_effect_with_suffix(
    content: str, suffix: str, filename: str, model_output: str
):
    original_suffix = suffix

    def _fn(prompt: str, suffix: str):
        assert original_suffix.startswith(suffix)
        assert (
            token_length(prompt) + token_length(suffix)
            <= PalmCodeGenBaseModel.MAX_MODEL_LEN
        )

        return TextGenModelOutput(text=model_output, score=-1)

    return _fn


def _side_effect_with_imports(
    content: str, suffix: str, filename: str, model_output: str
):
    def _fn(prompt: str, suffix: str):
        assert content.startswith("import os\nimport pytest")

        return TextGenModelOutput(text=model_output, score=-1)

    return _fn


def _side_effect_with_internal_exception(
    content: str, suffix: str, filename: str, model_output: str
):
    def _fn(prompt: str, suffix: str):
        raise VertexModelInternalError("internal error")

    return _fn


def _side_effect_with_invalid_arg_exception(
    content: str, suffix: str, filename: str, model_output: str
):
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
    _side_effect = model_gen_func(content, "", file_name, model_output)
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
    "language,prompt_builder_metadata,expected_completion,expected_prompt_symbol_counts",
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
                imports=MetadataExtraInfo(
                    name="imports",
                    pre=MetadataCodeContent(length=0, length_tokens=0),
                    post=MetadataCodeContent(length=0, length_tokens=0),
                ),
                function_signatures=MetadataExtraInfo(
                    name="function_signatures",
                    pre=MetadataCodeContent(length=0, length_tokens=0),
                    post=MetadataCodeContent(length=0, length_tokens=0),
                ),
            ),
            "random completion",
            None,
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
                imports=MetadataExtraInfo(
                    name="imports",
                    pre=MetadataCodeContent(length=0, length_tokens=0),
                    post=MetadataCodeContent(length=0, length_tokens=0),
                ),
                function_signatures=MetadataExtraInfo(
                    name="function_signatures",
                    pre=MetadataCodeContent(length=0, length_tokens=0),
                    post=MetadataCodeContent(length=0, length_tokens=0),
                ),
            ),
            "random completion\nnew line",
            None,
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
                imports=MetadataExtraInfo(
                    name="imports",
                    pre=MetadataCodeContent(length=0, length_tokens=0),
                    post=MetadataCodeContent(length=0, length_tokens=0),
                ),
                function_signatures=MetadataExtraInfo(
                    name="function_signatures",
                    pre=MetadataCodeContent(length=0, length_tokens=0),
                    post=MetadataCodeContent(length=0, length_tokens=0),
                ),
            ),
            "random completion\nnew line",
            {"comment": 1},
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
                imports=MetadataExtraInfo(
                    name="imports",
                    pre=MetadataCodeContent(length=22, length_tokens=5),
                    post=MetadataCodeContent(length=22, length_tokens=5),
                ),
                function_signatures=MetadataExtraInfo(
                    name="function_signatures",
                    pre=MetadataCodeContent(length=0, length_tokens=0),
                    post=MetadataCodeContent(length=0, length_tokens=0),
                ),
            ),
            "random completion\nnew line",
            {"comment": 1, "import_statement": 2},
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
            None,
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
            None,
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
    expected_prompt_symbol_counts,
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

    if prompt_builder_metadata:
        max_imports_len = int(
            text_gen_base_model.MAX_MODEL_LEN
            * ModelEnginePalm.MAX_TOKENS_IMPORTS_PERCENT
        )
        assert 0 <= completion.metadata.imports.post.length_tokens <= max_imports_len

        body_len = (
            text_gen_base_model.MAX_MODEL_LEN
            - completion.metadata.imports.post.length_tokens
        )
        max_suffix_len = int(body_len * ModelEnginePalm.MAX_TOKENS_SUFFIX_PERCENT)
        assert 0 <= completion.metadata.suffix.length_tokens <= max_suffix_len

        max_prefix_len = body_len - completion.metadata.suffix.length_tokens
        assert 0 <= completion.metadata.prefix.length_tokens <= max_prefix_len

    assert True if len(prefix) > 0 else len(prefix) == completion.metadata.prefix.length
    assert True if len(suffix) > 0 else len(suffix) == completion.metadata.suffix.length

    if expected_completion:
        engine.instrumentator.watcher.register_model_output_length.assert_called_with(
            model_output
        )
        engine.instrumentator.watcher.register_model_score.assert_called_with(-1)
    else:
        engine.instrumentator.watcher.register_model_output_length.assert_not_called
        engine.instrumentator.watcher.register_model_score.assert_not_called

    if expected_prompt_symbol_counts:
        engine.instrumentator.watcher.register_prompt_symbols.assert_called_with(
            expected_prompt_symbol_counts
        )
    else:
        engine.instrumentator.watcher.register_prompt_symbols.assert_not_called


@pytest.mark.parametrize(
    (
        "prefix",
        "suffix",
        "file_name",
        "lang_id",
        "expected_imports",
        "expected_functions",
    ),
    [
        (
            """
import numpy as np

def hello_world() -> int:
    return 1 + 4
""",
            """
def fib(n: int) -> int:
    # sh*t I forgot how to do fib!
    pass
""",
            "temp.py",
            ops.LanguageId.PYTHON,
            ["import numpy as np"],
            ["def hello_world() -> int:", "# def fib(n: int) -> int:"],
        ),
        (
            """
import numpy as np

def _side_effect_with_invalid_arg_exception(
    content: str, suffix: str, filename: str, model_output: str
):
    def _fn(prompt: str, suffix: str):
        raise VertexModelInvalidArgument(
""",
            """"invalid argument")

    return _fn

def fib(n: int) -> int:
    # sh*t I forgot how to do fib!
    pass
""",
            "temp.py",
            ops.LanguageId.PYTHON,
            ["import numpy as np"],
            [
                """def _side_effect_with_invalid_arg_exception(
    content: str, suffix: str, filename: str, model_output: str
):
""",
                "# def fib(n: int) -> int:",
            ],
        ),
    ],
)
def test_prompt_building_model_engine_palm(
    text_gen_base_model,
    prefix: str,
    suffix: str,
    file_name: str,
    lang_id: ops.LanguageId,
    expected_imports: list[str],
    expected_functions: list[str],
):
    engine = ModelEnginePalm(text_gen_base_model, tokenizer)
    prompt = engine._build_prompt(
        prefix=prefix, file_name=file_name, suffix=suffix, lang_id=lang_id
    )

    for expected_import in expected_imports:
        assert expected_import in prompt.prefix

    for expected_function in expected_functions:
        assert expected_function in prompt.prefix
