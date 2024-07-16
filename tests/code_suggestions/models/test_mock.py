import json

import pytest

from ai_gateway.models import (
    Message,
    ModelMetadata,
    Role,
    SafetyAttributes,
    TextGenModelOutput,
    mock,
)


@pytest.mark.asyncio
class TestLLM:
    TEST_CASES = [
        ("long custom prefix", "long custom suffix", {}),
        ("long custom prefix", "long custom suffix", {"temperature": 0.1}),
        ("long custom prefix", None, {}),
        ("long custom prefix", None, {"temperature": 0.1}),
    ]

    async def test_metadata(self):
        model = mock.LLM()
        assert model.metadata == ModelMetadata(
            name="llm-mocked", engine="llm-provider-mocked"
        )

    @pytest.mark.parametrize(("prefix", "suffix", "kwargs"), TEST_CASES)
    async def test_non_stream(self, prefix: str, suffix: str, kwargs: dict):
        model = mock.LLM()

        response = await model.generate(prefix, suffix, stream=False, **kwargs)
        assert isinstance(response, TextGenModelOutput)

        expected_substrings = [prefix, suffix] if suffix else [prefix]
        expected_substrings += [json.dumps(kwargs)]

        assert all(substring in response.text for substring in expected_substrings)
        assert response.text.startswith("echo:")
        assert response.score == 0
        assert response.safety_attributes == SafetyAttributes()

    @pytest.mark.parametrize(("prefix", "suffix", "kwargs"), TEST_CASES)
    async def test_stream(self, prefix: str, suffix: str, kwargs: dict):
        model = mock.LLM()

        response = await model.generate(prefix, suffix, stream=True, **kwargs)
        assert isinstance(response, mock.AsyncStream)

        actual_text = "".join([chunk.text async for chunk in response])
        expected_substrings = [prefix, suffix] if suffix else [prefix]
        expected_substrings += [json.dumps(kwargs)]

        assert all(substring in actual_text for substring in expected_substrings)


@pytest.mark.asyncio
class TestChatModel:
    TEST_CASES = [
        ([Message(role=Role.SYSTEM, content="long custom system prompt")], {}),
        (
            [
                Message(role=Role.SYSTEM, content="long custom system prompt"),
                Message(role=Role.USER, content="long custom user prompt"),
            ],
            {"temperature": 0.1},
        ),
    ]

    async def test_metadata(self):
        model = mock.ChatModel()
        assert model.metadata == ModelMetadata(
            name="chat-model-mocked", engine="chat-model-provider-mocked"
        )

    @pytest.mark.parametrize(("messages", "kwargs"), TEST_CASES)
    async def test_non_stream(self, messages: list[Message], kwargs: dict):
        model = mock.ChatModel()

        response = await model.generate(messages, stream=False, **kwargs)
        assert isinstance(response, TextGenModelOutput)

        messages = [message.model_dump(mode="json") for message in messages]
        expected_substrings = [json.dumps(s) for s in (messages, kwargs)]

        assert response.score == 0
        assert response.safety_attributes == SafetyAttributes()
        assert response.text.startswith("echo:")
        assert all(substring in response.text for substring in expected_substrings)

    @pytest.mark.parametrize(("messages", "kwargs"), TEST_CASES)
    async def test_stream(self, messages: list[Message], kwargs: dict):
        model = mock.ChatModel()

        response = await model.generate(messages, stream=True, **kwargs)
        assert isinstance(response, mock.AsyncStream)

        actual_text = "".join([chunk.text async for chunk in response])

        messages = [message.model_dump(mode="json") for message in messages]
        expected_substrings = [json.dumps(s) for s in (messages, kwargs)]

        assert actual_text.startswith("echo:")
        assert all(substring in actual_text for substring in expected_substrings)
