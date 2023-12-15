from typing import Sequence
from unittest.mock import AsyncMock, Mock

import pytest
from google.api_core.exceptions import InvalidArgument, RetryError
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient, PredictResponse
from google.protobuf import json_format

from ai_gateway.models.base import SafetyAttributes
from ai_gateway.models.palm import (
    CodeBisonModelInput,
    CodeGeckoModelInput,
    PalmCodeBisonModel,
    PalmCodeGeckoModel,
    PalmCodeGenBaseModel,
    PalmCodeGenModel,
    PalmModel,
    PalmTextBisonModel,
    TextBisonModelInput,
    TextGenModelOutput,
    VertexAPIConnectionError,
    VertexAPIStatusError,
)

TEST_PREFIX = "random prompt"
TEST_SUFFIX = "some suffix"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,prefix,suffix,expected_output,expected_generate_args",
    [
        (
            PalmTextBisonModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [TextBisonModelInput(TEST_PREFIX), 0.2, 32, 0.95, 40, None],
        ),
        (
            PalmTextBisonModel,
            "",
            TEST_SUFFIX,
            "",
            [TextBisonModelInput(""), 0.2, 32, 0.95, 40, None],
        ),
        (
            PalmCodeBisonModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [CodeBisonModelInput(TEST_PREFIX), 0.2, 2048, 0.95, 40, None],
        ),
        (
            PalmCodeBisonModel,
            "",
            TEST_SUFFIX,
            "",
            [CodeBisonModelInput(""), 0.2, 2048, 0.95, 40, None],
        ),
        (
            PalmCodeGeckoModel,
            TEST_PREFIX,
            TEST_SUFFIX,
            "some output",
            [
                CodeGeckoModelInput(TEST_PREFIX, TEST_SUFFIX),
                0.2,
                64,
                0.95,
                40,
                ["\n\n"],
            ],
        ),
        (
            PalmCodeGeckoModel,
            "",
            TEST_SUFFIX,
            "",
            [
                CodeGeckoModelInput("", TEST_SUFFIX),
                0.2,
                64,
                0.95,
                40,
                ["\n\n"],
            ],
        ),
    ],
)
async def test_palm_model_generate(
    model,
    prefix,
    suffix,
    expected_output,
    expected_generate_args,
):
    palm_model = model(client=Mock(), project="test", location="some location")
    palm_model._generate = AsyncMock(
        side_effect=lambda *_: TextGenModelOutput(
            text=expected_output, score=0, safety_attributes=SafetyAttributes()
        )
    )

    result = await palm_model.generate(prefix, suffix)

    assert result == TextGenModelOutput(
        text=expected_output, score=0, safety_attributes=SafetyAttributes()
    )

    palm_model._generate.assert_called_with(*expected_generate_args)


@pytest.mark.parametrize(
    "model_input,is_valid,output_dict",
    [
        (CodeBisonModelInput(TEST_PREFIX), True, {"prefix": TEST_PREFIX}),
        (CodeBisonModelInput(""), False, None),
        (TextBisonModelInput(TEST_PREFIX), True, {"content": TEST_PREFIX}),
        (TextBisonModelInput(""), False, None),
        (
            CodeGeckoModelInput(TEST_PREFIX, TEST_SUFFIX),
            True,
            {"prefix": TEST_PREFIX, "suffix": TEST_SUFFIX},
        ),
        (CodeGeckoModelInput("", ""), False, None),
        (CodeGeckoModelInput("", TEST_SUFFIX), False, None),
    ],
)
def test_palm_model_inputs(model_input, is_valid, output_dict):
    assert model_input.is_valid() is is_valid
    assert output_dict is None or model_input.dict() == output_dict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,client_exception,expected_exception",
    [
        (
            PalmCodeGeckoModel(
                Mock(spec=PredictionServiceAsyncClient),
                "random_project",
                "random_location",
            ),
            RetryError("Retry.", "Lost connection."),
            VertexAPIConnectionError,
        ),
        (
            PalmCodeGeckoModel(
                Mock(spec=PredictionServiceAsyncClient),
                "random_project",
                "random_location",
            ),
            InvalidArgument("Bad argument."),
            VertexAPIStatusError,
        ),
    ],
)
async def test_palm_model_api_error(model, client_exception, expected_exception):
    def _client_predict(*args, **kwargs):
        raise client_exception

    model.client.predict = AsyncMock(side_effect=_client_predict)

    with pytest.raises(expected_exception):
        result = await model.generate("random_prefix", "random_suffix")
        assert result == TextGenModelOutput(
            text="", score=0, safety_attributes=SafetyAttributes()
        )


@pytest.mark.parametrize(
    ("model_name_version", "expected_metadata_name"),
    [
        (PalmModel.TEXT_BISON.value, f"{PalmModel.TEXT_BISON.value}"),
        (PalmModel.CODE_BISON.value, f"{PalmModel.CODE_BISON.value}"),
        (PalmModel.CODE_GECKO.value, f"{PalmModel.CODE_GECKO.value}"),
        (f"{PalmModel.TEXT_BISON}@001", f"{PalmModel.TEXT_BISON}@001"),
        (f"{PalmModel.CODE_BISON}@001", f"{PalmModel.CODE_BISON}@001"),
        (f"{PalmModel.CODE_GECKO}@001", f"{PalmModel.CODE_GECKO}@001"),
    ],
)
def test_palm_model_from_name(
    model_name_version: str,
    expected_metadata_name: str,
):
    model = PalmCodeGenModel.from_model_name(
        model_name_version, Mock(), "project", "location"
    )

    assert isinstance(model.metadata.name, str)

    assert model.metadata.name == expected_metadata_name
    assert model.metadata.engine == "vertex-ai"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "stop_sequences", "expected_stop_sequences"),
    [
        (
            PalmTextBisonModel,
            None,
            None,
        ),
        (
            PalmCodeBisonModel,
            None,
            None,
        ),
        (
            PalmCodeGeckoModel,
            None,
            ["\n\n"],  # we set this sequence by default
        ),
        (
            PalmCodeGeckoModel,
            ["\n\n"],
            ["\n\n"],
        ),
        (
            PalmCodeGeckoModel,
            ["random stop sequence"],
            ["random stop sequence"],
        ),
    ],
)
async def test_palm_model_stop_sequences(
    model: PalmCodeGenBaseModel,
    stop_sequences: Sequence[str],
    expected_stop_sequences: Sequence[str],
):
    client = Mock()
    client.predict = AsyncMock(return_value=PredictResponse())
    palm_model = model(client=client, project="test", location="some location")

    await palm_model.generate("foo", "", stop_sequences=stop_sequences)

    client.predict.assert_called_once()

    parameters = client.predict.call_args[1]["parameters"]
    params_dict = json_format.MessageToDict(parameters)
    assert params_dict.get("stopSequences", None) == expected_stop_sequences


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "prediction", "expected_safety_attributes"),
    [
        (
            PalmTextBisonModel,
            {
                "safetyAttributes": {
                    "categories": ["Violent"],
                    "blocked": True,
                    "scores": [1.0],
                },
                "content": "",
            },
            SafetyAttributes(categories=["Violent"], blocked=True),
        ),
        (
            PalmCodeBisonModel,
            {
                "safetyAttributes": {
                    "categories": ["Violent"],
                    "blocked": True,
                    "scores": [1.0],
                },
                "content": "",
            },
            SafetyAttributes(categories=["Violent"], blocked=True),
        ),
        (
            PalmCodeGeckoModel,
            {
                "safetyAttributes": {
                    "categories": ["Violent"],
                    "blocked": True,
                    "scores": [1.0],
                },
                "content": "",
            },
            SafetyAttributes(categories=["Violent"], blocked=True),
        ),
        (
            PalmCodeGeckoModel,
            {
                "safetyAttributes": {
                    "errors": [234],
                    "blocked": True,
                },
                "content": "",
            },
            SafetyAttributes(errors=[234], blocked=True),
        ),
        (
            PalmCodeGeckoModel,
            {
                "safetyAttributes": {
                    "categories": [],
                    "blocked": False,
                    "scores": [],
                },
                "content": "def awesome_func",
            },
            SafetyAttributes(categories=[], blocked=False),
        ),
        (
            PalmCodeGeckoModel,
            {
                "content": "def awesome_func",
            },
            SafetyAttributes(categories=[], blocked=False),
        ),
    ],
)
async def test_palm_model_safety_attributes(
    model: PalmCodeGenBaseModel,
    prediction: dict,
    expected_safety_attributes: SafetyAttributes,
):
    client = Mock()
    predict_response = PredictResponse()
    predict_response.predictions.append(prediction)
    client.predict = AsyncMock(return_value=predict_response)
    palm_model = model(client=client, project="test", location="some location")

    model_output = await palm_model.generate("# bomberman", "")

    assert model_output.safety_attributes == expected_safety_attributes
