from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, PropertyMock

import pytest
from transformers import AutoTokenizer

from ai_gateway.code_suggestions.processing import (
    MetadataCodeContent,
    MetadataExtraInfo,
    MetadataPromptBuilder,
    ModelEngineCompletions,
    ops,
)
from ai_gateway.code_suggestions.processing.post.completions import PostProcessor
from ai_gateway.experimentation import ExperimentRegistry
from ai_gateway.models import (
    ModelMetadata,
    PalmCodeGenBaseModel,
    SafetyAttributes,
    TextGenModelOutput,
    VertexModelInternalError,
    VertexModelInvalidArgument,
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
    content: str,
    _suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str, _suffix: str):
        assert lang_id.name.lower() in prompt
        assert content in prompt

        return TextGenModelOutput(
            text=model_output, score=-1, safety_attributes=safety_attributes
        )

    return _fn


def _side_effect_unknown_tpl(
    content: str,
    _suffix: str,
    _: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    def _fn(prompt: str, _suffix: str):
        assert content == prompt

        return TextGenModelOutput(
            text=model_output, score=-1, safety_attributes=safety_attributes
        )

    return _fn


def _side_effect_unknown_tpl_palm(
    prefix: str,
    _suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    def _fn(prompt: str, _suffix: str):
        assert filename in prompt

        return TextGenModelOutput(
            text=model_output, score=-1, safety_attributes=safety_attributes
        )

    return _fn


def _side_effect_lang_prepended(
    content: str,
    _suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    lang_id = ops.lang_from_filename(filename)

    def _fn(prompt: str, _suffix: str):
        assert prompt.startswith(f"<{lang_id.name.lower()}>")
        assert prompt.endswith(content)

        return TextGenModelOutput(
            text=model_output, score=-1, safety_attributes=safety_attributes
        )

    return _fn


def _side_effect_with_suffix(
    content: str,
    suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    original_suffix = suffix

    def _fn(prompt: str, suffix: str):
        assert original_suffix.startswith(suffix)
        assert (
            token_length(prompt) + token_length(suffix)
            <= PalmCodeGenBaseModel.MAX_MODEL_LEN
        )

        return TextGenModelOutput(
            text=model_output, score=-1, safety_attributes=safety_attributes
        )

    return _fn


def _side_effect_with_imports(
    content: str,
    suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    def _fn(prompt: str, suffix: str):
        assert content.startswith("import os\nimport pytest")

        return TextGenModelOutput(
            text=model_output, score=-1, safety_attributes=safety_attributes
        )

    return _fn


def _side_effect_with_internal_exception(
    content: str,
    suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    def _fn(prompt: str, suffix: str):
        raise VertexModelInternalError("internal error")

    return _fn


def _side_effect_with_invalid_arg_exception(
    content: str,
    suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    def _fn(prompt: str, suffix: str):
        raise VertexModelInvalidArgument("invalid argument")

    return _fn


def token_length(s: str):
    return len(tokenizer(s)["input_ids"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix,suffix,file_name,editor_language,model_gen_func,model_output,safety_attributes,"
    "language,prompt_builder_metadata,expected_completion,expected_prompt_symbol_counts,expected_safety_attributes",
    [
        (
            "prompt",
            "",
            "f.unk",
            None,
            _side_effect_unknown_tpl_palm,
            "random completion",
            SafetyAttributes(),
            None,
            MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=6, length_tokens=2),
                    "suffix": MetadataCodeContent(length=0, length_tokens=0),
                },
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
            SafetyAttributes(),
        ),
        (
            "prompt",
            "",
            "f.unk",
            None,
            _side_effect_unknown_tpl_palm,
            "random completion",
            SafetyAttributes(),
            None,
            MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=6, length_tokens=2),
                    "suffix": MetadataCodeContent(length=0, length_tokens=0),
                },
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
            SafetyAttributes(),
        ),
        (
            "prompt",
            "",
            "f.unk",
            "typescript",
            _side_effect_unknown_tpl_palm,
            "random completion",
            SafetyAttributes(),
            ops.LanguageId.TS,
            MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=6, length_tokens=2),
                    "suffix": MetadataCodeContent(length=0, length_tokens=0),
                },
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
            SafetyAttributes(),
        ),
        (
            "prompt",
            "",
            "f.unk",
            None,
            _side_effect_unknown_tpl_palm,
            "random completion\nnew line",
            SafetyAttributes(),
            None,
            MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=6, length_tokens=2),
                    "suffix": MetadataCodeContent(length=0, length_tokens=0),
                },
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
            SafetyAttributes(),
        ),
        (
            "prompt " * 2048,
            "abc " * 4096,
            "f.py",
            None,
            _side_effect_with_suffix,
            "random completion\nnew line",
            SafetyAttributes(),
            ops.LanguageId.PYTHON,
            MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=3494, length_tokens=500),
                    "suffix": MetadataCodeContent(length=1002, length_tokens=500),
                },
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
            SafetyAttributes(),
        ),
        (
            "import os\nimport pytest\n" + "prompt" * 2048,
            "",
            "f.py",
            None,
            _side_effect_with_imports,
            "random completion",
            SafetyAttributes(),
            ops.LanguageId.PYTHON,
            MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=2984, length_tokens=995),
                    "suffix": MetadataCodeContent(length=0, length_tokens=0),
                },
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
            "random completion",
            {"comment": 1, "import_statement": 2},
            SafetyAttributes(),
        ),
        (
            "random_prefix",
            "random_suffix",
            "f.unk",
            "",
            _side_effect_with_internal_exception,
            "unreturned completion due to an exception",
            SafetyAttributes(),
            None,
            None,
            "",
            None,
            None,
        ),
        (
            "random_prefix",
            "random_suffix",
            "f.unk",
            "",
            _side_effect_with_invalid_arg_exception,
            "unreturned completion due to an exception",
            SafetyAttributes(),
            None,
            None,
            "",
            None,
            None,
        ),
        (
            "",
            "",
            "app.js",
            "",
            _side_effect_with_suffix,
            "",
            SafetyAttributes(),
            ops.LanguageId.JS,
            None,
            "",
            None,
            SafetyAttributes(),
        ),
    ],
)
async def test_model_engine_palm(
    text_gen_base_model,
    prefix,
    suffix,
    file_name,
    editor_language,
    model_gen_func,
    model_output,
    safety_attributes,
    language,
    prompt_builder_metadata,
    expected_completion,
    expected_prompt_symbol_counts,
    expected_safety_attributes,
):
    model_name = "palm-model"
    model_engine = "vertex-ai"
    _side_effect = model_gen_func(
        prefix, suffix, file_name, model_output, safety_attributes
    )
    _model_metadata = ModelMetadata(name=model_name, engine=model_engine)

    text_gen_base_model.generate = AsyncMock(side_effect=_side_effect)
    type(text_gen_base_model).metadata = PropertyMock(return_value=_model_metadata)

    engine = ModelEngineCompletions(
        model=text_gen_base_model,
        tokenizer=tokenizer,
        post_processor=PostProcessor,
        experiment_registry=ExperimentRegistry(),
    )
    engine.instrumentator = MockInstrumentor()
    completion = await engine.generate(prefix, suffix, file_name, editor_language)

    assert completion.text == expected_completion
    assert completion.model == _model_metadata
    assert completion.lang_id == language

    if prompt_builder_metadata:
        max_imports_len = int(
            text_gen_base_model.MAX_MODEL_LEN
            * ModelEngineCompletions.MAX_TOKENS_IMPORTS_PERCENT
        )
        assert 0 <= completion.metadata.imports.post.length_tokens <= max_imports_len

        components = completion.metadata.components
        body_len = (
            text_gen_base_model.MAX_MODEL_LEN
            - completion.metadata.imports.post.length_tokens
        )
        max_suffix_len = int(
            body_len * ModelEngineCompletions.MAX_TOKENS_SUFFIX_PERCENT
        )
        assert 0 <= components["suffix"].length_tokens <= max_suffix_len

        max_prefix_len = body_len - components["suffix"].length_tokens
        assert 0 <= components["prefix"].length_tokens <= max_prefix_len

    if not prefix and completion.metadata:
        assert 0 == completion.metadata.components["prefix"].length
        assert 0 == completion.metadata.components["suffix"].length

    watcher = engine.instrumentator.watcher
    if expected_completion:
        watcher.register_model_output_length.assert_called_with(model_output)
        watcher.register_model_score.assert_called_with(-1)
        watcher.register_safety_attributes.assert_called_with(
            expected_safety_attributes
        )
    else:
        watcher.register_model_output_length.assert_not_called
        watcher.register_model_score.assert_not_called

    if expected_prompt_symbol_counts:
        watcher.register_prompt_symbols.assert_called_with(
            expected_prompt_symbol_counts
        )
    else:
        watcher.register_prompt_symbols.assert_not_called

    if expected_safety_attributes:
        watcher.register_safety_attributes.assert_called_with(
            expected_safety_attributes
        )
    else:
        watcher.register_safety_attributes.assert_not_called()


JAVASCRIPT_SOURCE_SAMPLE = """
import React, { useState } from "react";

const App = () => {
  const [date, setDate] = useState(new Date());
  const [number, setNumber] = useState(0);

  const addNumber = () => {
    setNumber(sum(number, 1));
  };

  const getDateString = () => {
    return dateFns.format(date, "YYYY-MM-DD");
  };

  return (
    <div>
      <h1>Date: {getDateString()}</h1>
      <h1>Number: {number}</h1>
      <button onClick={addNumber}>Add 1</button>
    </div>
  );
};

export default App;
"""


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
        (
            JAVASCRIPT_SOURCE_SAMPLE[:50],
            JAVASCRIPT_SOURCE_SAMPLE[50:],
            "app.js",
            ops.LanguageId.JS,
            ['import React, { useState } from "react"'],
            [],
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
    engine = ModelEngineCompletions(
        model=text_gen_base_model,
        tokenizer=tokenizer,
        post_processor=PostProcessor,
        experiment_registry=ExperimentRegistry(),
    )
    prompt = engine._build_prompt(
        prefix=prefix, file_name=file_name, suffix=suffix, lang_id=lang_id
    )

    for expected_import in expected_imports:
        assert expected_import in prompt.prefix

    for expected_function in expected_functions:
        assert expected_function in prompt.prefix
