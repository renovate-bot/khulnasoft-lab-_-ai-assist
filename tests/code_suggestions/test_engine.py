from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, PropertyMock

import pytest
from google.api_core.exceptions import GoogleAPICallError, GoogleAPIError
from transformers import AutoTokenizer

from ai_gateway.code_suggestions.processing import (
    MetadataCodeContent,
    MetadataExtraInfo,
    MetadataPromptBuilder,
    ModelEngineCompletions,
    ops,
)
from ai_gateway.code_suggestions.processing.pre import TokenizerTokenStrategy
from ai_gateway.experimentation import ExperimentRegistry
from ai_gateway.models import (
    ModelAPIError,
    ModelMetadata,
    PalmCodeGenBaseModel,
    SafetyAttributes,
    TextGenModelOutput,
    VertexAPIConnectionError,
    VertexAPIStatusError,
)
from ai_gateway.models.base import TokensConsumptionMetadata

tokenization_strategy = TokenizerTokenStrategy(
    tokenizer=AutoTokenizer.from_pretrained("Salesforce/codegen2-16B")
)


class MockInstrumentor:
    def __init__(self):
        self.watcher = Mock()
        self.watcher.register_prompt_symbols = Mock()
        self.watcher.register_model_output_length = Mock()
        self.watcher.register_lang = Mock()

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

        return [
            TextGenModelOutput(
                text=model_output, score=-1, safety_attributes=safety_attributes
            )
        ]

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
            tokenization_strategy.estimate_length(prompt)[0]
            + tokenization_strategy.estimate_length(suffix)[0]
            <= PalmCodeGenBaseModel.MAX_MODEL_LEN
        )

        return [
            TextGenModelOutput(
                text=model_output, score=-1, safety_attributes=safety_attributes
            )
        ]

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

        return [
            TextGenModelOutput(
                text=model_output, score=-1, safety_attributes=safety_attributes
            )
        ]

    return _fn


def _side_effect_with_tokens_consumption_metadata(
    content: str,
    suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    def _fn(prompt: str, suffix: str):
        return [
            TextGenModelOutput(
                text=model_output,
                score=-1,
                safety_attributes=safety_attributes,
                metadata=TokensConsumptionMetadata(input_tokens=1, output_tokens=2),
            )
        ]

    return _fn


def _side_effect_with_connection_exception(
    content: str,
    suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    def _fn(prompt: str, suffix: str):
        VertexAPIConnectionError.code = -1
        raise VertexAPIConnectionError("connection exception")

    return _fn


def _side_effect_with_status_exception(
    content: str,
    suffix: str,
    filename: str,
    model_output: str,
    safety_attributes: SafetyAttributes,
):
    def _fn(prompt: str, suffix: str):
        VertexAPIStatusError.code = 404
        raise VertexAPIStatusError("status exception")

    return _fn


@contextmanager
def does_not_raise():
    yield


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix,suffix,file_name,editor_language,model_gen_func,successful_predict,model_output,safety_attributes,"
    "language,prompt_builder_metadata,expected_completion,expected_prompt_symbol_counts,expected_safety_attributes,"
    "expected_input_tokens,expected_output_tokens,estimate_tokens_consumption",
    [
        (
            "prompt",
            "",
            "f.unk",
            None,
            _side_effect_unknown_tpl_palm,
            True,
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
            None,
            None,
            True,
        ),
        (
            "prompt",
            "",
            "f.unk",
            None,
            _side_effect_with_tokens_consumption_metadata,
            True,
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
            1,
            2,
            False,
        ),
        (
            "prompt",
            "",
            "f.unk",
            None,
            _side_effect_unknown_tpl_palm,
            True,
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
            None,
            None,
            False,
        ),
        (
            "prompt",
            "",
            "f.unk",
            None,
            _side_effect_unknown_tpl_palm,
            True,
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
            None,
            None,
            False,
        ),
        (
            "prompt",
            "",
            "f.unk",
            "typescript",
            _side_effect_unknown_tpl_palm,
            True,
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
            {"comment": 1},
            SafetyAttributes(),
            None,
            None,
            False,
        ),
        (
            "prompt",
            "",
            "f.unk",
            None,
            _side_effect_unknown_tpl_palm,
            True,
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
            None,
            None,
            False,
        ),
        (
            "prompt " * 2048,
            "abc " * 4096,
            "f.py",
            None,
            _side_effect_with_suffix,
            True,
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
            None,
            None,
            False,
        ),
        (
            "prompt " * 2048,
            "abc " * 4096,
            "f.vue",
            None,
            _side_effect_with_suffix,
            True,
            "random completion\nnew line",
            SafetyAttributes(),
            None,
            MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=3494, length_tokens=500),
                    "suffix": MetadataCodeContent(length=1002, length_tokens=500),
                },
            ),
            "random completion\nnew line",
            None,
            SafetyAttributes(),
            None,
            None,
            False,
        ),
        (
            "import os\nimport pytest\n" + "prompt" * 2048,
            "",
            "f.py",
            None,
            _side_effect_with_imports,
            True,
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
            None,
            None,
            False,
        ),
        (
            "random_prefix",
            "random_suffix",
            "f.unk",
            "",
            _side_effect_with_connection_exception,
            False,
            "unreturned completion due to an exception",
            SafetyAttributes(),
            None,
            None,
            "",
            None,
            None,
            None,
            None,
            False,
        ),
        (
            "random_prefix",
            "random_suffix",
            "f.unk",
            "",
            _side_effect_with_status_exception,
            False,
            "unreturned completion due to an exception",
            SafetyAttributes(),
            None,
            None,
            "",
            None,
            None,
            None,
            None,
            False,
        ),
        (
            "",
            "",
            "app.js",
            "",
            _side_effect_with_suffix,
            True,
            "",
            SafetyAttributes(),
            ops.LanguageId.JS,
            None,
            "",
            {"comment": 1},
            SafetyAttributes(),
            None,
            None,
            False,
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
    successful_predict,
    model_output,
    safety_attributes,
    language,
    prompt_builder_metadata,
    expected_completion,
    expected_prompt_symbol_counts,
    expected_safety_attributes,
    expected_input_tokens,
    expected_output_tokens,
    estimate_tokens_consumption,
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
        tokenization_strategy=tokenization_strategy,
        experiment_registry=ExperimentRegistry(),
    )
    engine.instrumentator = MockInstrumentor()

    if successful_predict:
        raises_expectation = does_not_raise()
    else:
        raises_expectation = pytest.raises(ModelAPIError)

    with raises_expectation:
        completion = (
            await engine.generate(prefix, suffix, file_name, editor_language)
        )[0]

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

    if estimate_tokens_consumption:
        assert completion.tokens_consumption_metadata.output_tokens == (
            tokenization_strategy.estimate_length(model_output)[0]
        )
        assert completion.tokens_consumption_metadata.input_tokens == (
            completion.metadata.components["prefix"].length_tokens
            + completion.metadata.components["suffix"].length_tokens
        )

    if expected_input_tokens is not None and expected_output_tokens is not None:
        assert completion.tokens_consumption_metadata.output_tokens == (
            expected_output_tokens
        )
        assert completion.tokens_consumption_metadata.input_tokens == (
            expected_input_tokens
        )

    if not prefix and completion.metadata:
        assert 0 == completion.metadata.components["prefix"].length
        assert 0 == completion.metadata.components["suffix"].length

    watcher = engine.instrumentator.watcher
    watcher.register_lang.assert_called_with(language, editor_language)

    if successful_predict:
        watcher.register_model_output_length.assert_called_with(model_output)
        watcher.register_model_score.assert_called_with(-1)
        assert completion.text == expected_completion
        assert completion.model == _model_metadata
        assert completion.lang_id == language
    else:
        watcher.register_model_output_length.assert_not_called()
        watcher.register_model_score.assert_not_called()

    if expected_prompt_symbol_counts:
        watcher.register_prompt_symbols.assert_called_with(
            expected_prompt_symbol_counts
        )
    else:
        watcher.register_prompt_symbols.assert_not_called()

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
        "code_context",
        "expected_imports",
        "expected_functions",
        "expected_context",
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
            ["import pytest"],
            ["import numpy as np"],
            ["def hello_world() -> int:", "# def fib(n: int) -> int:"],
            ["import pytest"],
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
            [],
            ["import numpy as np"],
            [
                """def _side_effect_with_invalid_arg_exception(
    content: str, suffix: str, filename: str, model_output: str
):
""",
                "# def fib(n: int) -> int:",
            ],
            [],
        ),
        (
            JAVASCRIPT_SOURCE_SAMPLE[:50],
            JAVASCRIPT_SOURCE_SAMPLE[50:],
            "app.js",
            ops.LanguageId.JS,
            [],
            ['import React, { useState } from "react"'],
            [],
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_prompt_building_model_engine_palm(
    text_gen_base_model,
    prefix: str,
    suffix: str,
    file_name: str,
    lang_id: ops.LanguageId,
    code_context: list[str],
    expected_imports: list[str],
    expected_functions: list[str],
    expected_context: list[str],
):
    engine = ModelEngineCompletions(
        model=text_gen_base_model,
        tokenization_strategy=tokenization_strategy,
        experiment_registry=ExperimentRegistry(),
    )
    prompt = await engine._build_prompt(
        prefix=prefix,
        file_name=file_name,
        suffix=suffix,
        lang_id=lang_id,
        code_context=code_context,
    )

    for expected_import in expected_imports:
        assert expected_import in prompt.prefix

    for expected_function in expected_functions:
        assert expected_function in prompt.prefix

    for expected_context in expected_context:
        assert expected_context in prompt.prefix
