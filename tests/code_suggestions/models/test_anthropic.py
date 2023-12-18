from typing import Type
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    BadRequestError,
    UnprocessableEntityError,
)
from anthropic.resources import AsyncCompletions
from anthropic.types import Completion

from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicModel,
    SafetyAttributes,
    TextGenModelChunk,
    TextGenModelOutput,
)


@pytest.mark.parametrize(
    "model_name_version",
    ["claude-instant-1.2", "claude-2.0"],
)
def test_anthropic_model_from_name(model_name_version: str):
    model = AnthropicModel.from_model_name(model_name_version, Mock())

    assert model.metadata.name == model_name_version
    assert model.metadata.engine == "anthropic"


@pytest.mark.parametrize(
    ("model_name_version", "opts", "opts_client", "opts_model"),
    [
        (
            "claude-instant-1.2",
            {},
            AnthropicModel.OPTS_CLIENT,
            AnthropicModel.OPTS_MODEL,
        ),
        (
            "claude-instant-1.2",
            {"version": "2020-10-10"},
            AnthropicModel.OPTS_CLIENT,
            AnthropicModel.OPTS_MODEL,
        ),
        (
            "claude-instant-1.2",
            {
                "timeout": 6,
                "max_tokens_to_sample": 5,
                "stop_sequences": ["\n\n"],
                "temperature": 0.1,
                "top_k": 40,
                "top_p": 0.95,
                "version": "2020-10-10",
                "default_headers": {
                    "Custom Header": "custom",
                    "anthropic-version": "2010-10-10",
                },
                "max_retries": 2,
            },
            {
                "default_headers": {
                    "Custom Header": "custom",
                    "anthropic-version": "2010-10-10",
                },
                "max_retries": 2,
            },
            {
                "timeout": 6,
                "max_tokens_to_sample": 5,
                "stop_sequences": ["\n\n"],
                "temperature": 0.1,
                "top_k": 40,
                "top_p": 0.95,
            },
        ),
    ],
)
def test_anthropic_provider_opts(
    model_name_version: str,
    opts: dict,
    opts_client: dict,
    opts_model: dict,
):
    client = Mock()
    model = AnthropicModel.from_model_name(model_name_version, client, **opts)

    headers = opts_client["default_headers"]
    if not headers.get("anthropic-version", None):
        headers["anthropic-version"] = opts.get(
            "version", AnthropicModel.DEFAULT_VERSION
        )

    client.with_options.assert_called_with(**opts_client)
    assert model.model_opts == opts_model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name_version", "exception", "expected_exception"),
    [
        ("claude-instant-1.2", BadRequestError, AnthropicAPIStatusError),
        ("claude-instant-1.2", UnprocessableEntityError, AnthropicAPIStatusError),
        ("claude-instant-1.2", APIConnectionError, AnthropicAPIConnectionError),
        ("claude-instant-1.2", APITimeoutError, AnthropicAPIConnectionError),
    ],
)
async def test_anthropic_model_error(
    model_name_version: str, exception: Type, expected_exception: Type
):
    def _client_predict(*_args, **_kwargs):
        if issubclass(exception, APITimeoutError):
            raise exception(request=Mock())

        if issubclass(exception, APIConnectionError):
            raise exception(message="exception", request=Mock())

        raise exception(message="exception", response=Mock(), body=Mock())

    model = AnthropicModel.from_model_name(
        model_name_version, Mock(spec=AsyncAnthropic)
    )
    model.client.completions.create = AsyncMock(side_effect=_client_predict)

    with pytest.raises(expected_exception):
        _ = await model.generate("prefix", "suffix")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "model_name_version",
        "prompt",
        "suggestion",
        "opts",
        "opts_model",
        "expected_opts_model",
        "expected_output",
    ),
    [
        (
            "claude-instant-1.2",
            "random_prompt",
            "random_text",
            {},
            {},
            {**AnthropicModel.OPTS_MODEL, **{"stream": False}},
            TextGenModelOutput(
                text="random_text", score=10_000, safety_attributes=SafetyAttributes()
            ),
        ),
        (
            "claude-instant-1.2",
            "random_prompt",
            "random_text",
            {"top_k": 10},
            {},
            {**AnthropicModel.OPTS_MODEL, **{"top_k": 10, "stream": False}},
            TextGenModelOutput(
                text="random_text", score=10_000, safety_attributes=SafetyAttributes()
            ),
        ),
        (
            "claude-instant-1.2",
            "random_prompt",
            "random_text",
            {"temperature": 1},
            {},
            {**AnthropicModel.OPTS_MODEL, **{"temperature": 1, "stream": False}},
            TextGenModelOutput(
                text="random_text", score=10_000, safety_attributes=SafetyAttributes()
            ),
        ),
        (
            "claude-instant-1.2",
            "random_prompt",
            "random_text",
            {"temperature": 1},
            {"temperature": 0.1},  # Override the temperature when calling the model
            {**AnthropicModel.OPTS_MODEL, **{"temperature": 0.1, "stream": False}},
            TextGenModelOutput(
                text="random_text", score=10_000, safety_attributes=SafetyAttributes()
            ),
        ),
        (
            "claude-instant-1.2",
            "random_prompt",
            "random_text",
            {},
            {
                "temperature": 0.1,
                "top_p": 0.95,
            },  # Override the temperature when calling the model
            {
                **AnthropicModel.OPTS_MODEL,
                **{"temperature": 0.1, "top_p": 0.95, "stream": False},
            },
            TextGenModelOutput(
                text="random_text", score=10_000, safety_attributes=SafetyAttributes()
            ),
        ),
    ],
)
async def test_anthropic_model_generate(
    model_name_version: str,
    prompt: str,
    suggestion: str,
    opts: dict,
    opts_model: dict,
    expected_opts_model: dict,
    expected_output: TextGenModelOutput,
):
    def _client_predict(*_args, **_kwargs):
        return Completion(
            completion=suggestion,
            model=model_name_version,
            stop_reason="max_tokens",
        )

    model = AnthropicModel.from_model_name(
        model_name_version,
        Mock(spec=AsyncAnthropic),
        **opts,
    )

    model.client.completions.create = AsyncMock(side_effect=_client_predict)

    actual_output = await model.generate(prompt, "", **opts_model)
    # Create another dictionary to avoid modifying the original one.
    expected_opts_model = {
        **expected_opts_model,
        **{
            "model": model_name_version,
            "prompt": prompt,
        },
    }

    model.client.completions.create.assert_called_with(**expected_opts_model)
    assert actual_output.text == expected_output.text


@pytest.mark.asyncio
async def test_anthropic_model_generate_instrumented():
    model = AnthropicModel(
        model_name=AnthropicModel.CLAUDE_V2_0, client=Mock(spec=AsyncAnthropic)
    )
    model.client.completions.create = AsyncMock()

    with patch(
        "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
    ) as mock_watch:
        await model.generate("Wolf, what time is it?")

        mock_watch.assert_called_once_with(stream=False)


@pytest.mark.asyncio
async def test_anthropic_model_generate_stream_instrumented():
    async def mock_stream(*args, **kwargs):
        completions = [
            Completion(
                completion="hello", model=AnthropicModel.CLAUDE_V2_0, stop_reason=""
            ),
            Completion(
                completion="world", model=AnthropicModel.CLAUDE_V2_0, stop_reason=""
            ),
            "break here",
        ]
        for item in completions:
            if item == "break here":
                raise ValueError("broken")
            yield item

    model = AnthropicModel(
        model_name=AnthropicModel.CLAUDE_V2_0, client=Mock(spec=AsyncAnthropic)
    )
    model.client.completions.create = AsyncMock(side_effect=mock_stream)

    with patch(
        "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
    ) as mock_watch:
        watcher = Mock()
        mock_watch.return_value.__enter__.return_value = watcher

        r = await model.generate("Wolf, what time is it?", stream=True)

        # Make sure we haven't called finish before completions are consumed
        watcher.finish.assert_not_called()

        with pytest.raises(ValueError):
            all_completions = [item async for item in r]

        mock_watch.assert_called_once_with(stream=True)
        watcher.finish.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "model_name_version",
        "prompt",
        "completion_chunks",
        "expected_chunks",
    ),
    [
        (
            "claude-instant-1.2",
            "random_prompt",
            [
                Completion(
                    completion="def hello_",
                    stop_reason="",
                    model="claude-instant-1.2",
                ),
                Completion(
                    completion="world():",
                    stop_reason="",
                    model="claude-instant-1.2",
                ),
            ],
            [
                "def hello_",
                "world():",
            ],
        ),
    ],
)
async def test_anthropic_model_generate_stream(
    model_name_version: str,
    prompt: str,
    completion_chunks: list[Completion],
    expected_chunks: list[str],
):
    async def _stream_generator(*args, **kwargs):
        for chunk in completion_chunks:
            yield chunk

    model = AnthropicModel.from_model_name(
        model_name_version,
        Mock(spec=AsyncAnthropic),
    )

    model.client.completions.create = AsyncMock(side_effect=_stream_generator)

    actual_output = await model.generate(prompt, stream=True)
    expected_opts_model = {
        **AnthropicModel.OPTS_MODEL,
        **{"model": model_name_version, "prompt": prompt, "stream": True},
    }

    chunks = []
    async for content in actual_output:
        chunks += content

    assert chunks == expected_chunks

    model.client.completions.create.assert_called_with(**expected_opts_model)
