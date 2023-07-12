import pytest

from contextlib import contextmanager
from unittest.mock import Mock
from codesuggestions.models.palm import (
    CodeBisonModelInput,
    CodeGeckoModelInput,
    PalmCodeBisonModel,
    PalmCodeGeckoModel,
    PalmTextBisonModel,
    TextBisonModelInput,
    TextGenModelOutput,
)
from typing import Any

from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.api_core.exceptions import InvalidArgument, InternalServerError


class MockInstrumentor:
    @contextmanager
    def watch(self, prompt: str, **kwargs: Any):
        yield Mock()


TEST_PREFIX = "random propmt"
TEST_SUFFIX = "some suffix"


@pytest.mark.parametrize(
    "model,prefix,suffix,expected_output,expected_generate_args",
    [
        (
            PalmTextBisonModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [TextBisonModelInput(TEST_PREFIX), 0.2, 32, 0.95, 40]
        ),
        (
            PalmTextBisonModel,
            "",
            TEST_SUFFIX,
            "",
            None
        ),
        (
            PalmCodeBisonModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [CodeBisonModelInput(TEST_PREFIX), 0.2, 32, 0.95, 40]
        ),
        (
            PalmCodeBisonModel,
            "",
            TEST_SUFFIX,
            "",
            None
        ),
        (
            PalmCodeGeckoModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [CodeGeckoModelInput(TEST_PREFIX, TEST_SUFFIX), 0.2, 32, 0.95, 40]
        ),
        (
            PalmCodeGeckoModel,
            "",
            TEST_SUFFIX,
            "",
            None
        )
    ])
def test_palm_code_gecko_prompt(model, prefix, suffix, expected_output, expected_generate_args):
    client = Mock()
    palm_model = model(client=client, project="test", location="some location")
    palm_model.instrumentator = MockInstrumentor()
    palm_model._generate = Mock(side_effect=lambda *_: TextGenModelOutput(text=expected_output))

    result = palm_model.generate(prefix, suffix)

    assert result == TextGenModelOutput(text=expected_output)

    if expected_generate_args:
        palm_model._generate.assert_called_with(*expected_generate_args)
        assert client.assert_called_once
    else:
        assert not client.called


@pytest.mark.parametrize(
    "model_input,is_valid,output_dict", [
        (CodeBisonModelInput(TEST_PREFIX), True, {"prefix": TEST_PREFIX}),
        (CodeBisonModelInput(""), False, None),
        (TextBisonModelInput(TEST_PREFIX), True, {"content": TEST_PREFIX}),
        (TextBisonModelInput(""), False, None),
        (CodeGeckoModelInput(TEST_PREFIX, TEST_SUFFIX), True, {"prefix": TEST_PREFIX, "suffix": TEST_SUFFIX}),
        (CodeGeckoModelInput("", ""), False, None),
        (CodeGeckoModelInput("", TEST_SUFFIX), False, None),
    ]
)
def test_palm_model_inputs(model_input, is_valid, output_dict):
    assert model_input.is_valid() is is_valid
    assert output_dict is None or model_input.dict() == output_dict


@pytest.mark.parametrize(
    "model,vertex_exception", [
        (
            PalmCodeGeckoModel(Mock(spec=PredictionServiceClient), "random_project", "random_location"),
            InvalidArgument("Bad argument."),
        ),
        (
            PalmCodeGeckoModel(Mock(spec=PredictionServiceClient), "random_project", "random_location"),
            InternalServerError("Internal server error."),
        ),
    ]
)
def test_palm_model_api_error(model, vertex_exception):
    def _client_predict(*args, **kwargs):
        raise vertex_exception

    model.client.predict = Mock(side_effect=_client_predict)
    model.instrumentator = MockInstrumentor()

    actual = model.generate("random_prefix", "random_suffix")
    assert actual == TextGenModelOutput(text="")
