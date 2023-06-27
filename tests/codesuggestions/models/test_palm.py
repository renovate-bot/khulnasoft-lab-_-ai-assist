import pytest

from contextlib import contextmanager
from unittest.mock import Mock
from codesuggestions.models.palm import PalmTextBisonModel, PalmCodeBisonModel, PalmCodeGeckoModel, TextGenModelOutput
from typing import Any


class MockInstrumentor:
    @contextmanager
    def watch(self, prompt: str, **kwargs: Any):
        yield


@pytest.mark.parametrize(
    "model,prefix,suffix,expected_output,expected_generate_args",
    [
        (
            PalmTextBisonModel,
            "random prompt",
            "some suffix",
            "some output",
            [{"content": "random prompt"}, 0.2, 32, 0.95, 40]
        ),
        (
            PalmTextBisonModel,
            "",
            "some suffix",
            "",
            None
        ),
        (
            PalmCodeBisonModel,
            "random prompt",
            "some suffix",
            "some output",
            [{"prefix": "random prompt"}, 0.2, 32, 0.95, 40]
        ),
        (
            PalmCodeBisonModel,
            "",
            "some suffix",
            "",
            None
        ),
        (
            PalmCodeGeckoModel,
            "random prompt",
            "some suffix",
            "some output",
            [{"prefix": "random prompt", "suffix": "some suffix"}, 0.2, 32, 0.95, 40]
        ),
        (
            PalmCodeGeckoModel,
            "",
            "some suffix",
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
    else:
        assert not palm_model._generate.called
