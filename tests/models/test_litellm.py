from typing import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_gateway.models import KindLiteLlmModel, LiteLlmChatModel
from ai_gateway.models.base import TextGenModelChunk, TextGenModelOutput
from ai_gateway.models.base_chat import Message


class TestKindLiteLlmModel:
    def test_chat_model(self):
        assert KindLiteLlmModel.MISTRAL.chat_model() == "openai/mistral"
        assert KindLiteLlmModel.MIXTRAL.chat_model() == "openai/mixtral"


class TestLiteLlmChatMode:
    @pytest.fixture
    def endpoint(self):
        return "http://127.0.0.1:1111/v1"

    @pytest.fixture
    def lite_llm_chat_model(self, endpoint):
        return LiteLlmChatModel.from_model_name(name="mistral", endpoint=endpoint)

    @pytest.mark.parametrize("model_name", ["mistral", "mixtral"])
    def test_from_model_name(self, model_name: str, endpoint):
        model = LiteLlmChatModel.from_model_name(name=model_name, endpoint=endpoint)

        assert model.metadata.name == f"openai/{model_name}"
        assert model.endpoint == endpoint
        assert model.metadata.engine == "litellm"

    @pytest.mark.asyncio
    async def test_generate(self, lite_llm_chat_model, endpoint):
        expected_messages = [{"role": "user", "content": "Test message"}]

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
            mock_acompletion.return_value = AsyncMock(
                choices=[AsyncMock(message=AsyncMock(content="Test response"))]
            )
            messages = [Message(content="Test message", role="user")]
            output = await lite_llm_chat_model.generate(messages)
            assert isinstance(output, TextGenModelOutput)
            assert output.text == "Test response"

            mock_acompletion.assert_called_with(
                lite_llm_chat_model.metadata.name,
                messages=expected_messages,
                stream=False,
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_tokens=2048,
                api_key="stubbed-api-key",
                api_base=endpoint,
                timeout=30.0,
                stop=["</new_code>"],
            )

    @pytest.mark.asyncio
    async def test_generate_stream(self, lite_llm_chat_model, endpoint):
        expected_messages = [{"role": "user", "content": "Test message"}]

        streamed_response = AsyncMock()
        streamed_response.__aiter__.return_value = iter(
            [
                AsyncMock(
                    choices=[AsyncMock(delta=AsyncMock(content="Streamed content"))]
                )
            ]
        )

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion, patch(
            "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
        ) as mock_watch:
            watcher = Mock()
            mock_watch.return_value.__enter__.return_value = watcher

            mock_acompletion.return_value = streamed_response

            messages = [Message(content="Test message", role="user")]
            response = await lite_llm_chat_model.generate(
                messages=messages,
                stream=True,
                temperature=0.3,
                top_p=0.9,
                top_k=25,
                max_output_tokens=1024,
            )

            content = []
            async for chunk in response:
                content.append(chunk.text)
            assert content == ["Streamed content"]

            mock_acompletion.assert_called_with(
                lite_llm_chat_model.metadata.name,
                messages=expected_messages,
                stream=True,
                temperature=0.3,
                top_p=0.9,
                top_k=25,
                max_tokens=1024,
                api_key="stubbed-api-key",
                api_base=endpoint,
                timeout=30.0,
                stop=["</new_code>"],
            )

            mock_watch.assert_called_once_with(stream=True)
            watcher.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_instrumented(self, lite_llm_chat_model, endpoint):
        async def mock_stream(*args, **kwargs):
            completions = [
                AsyncMock(
                    choices=[AsyncMock(delta=AsyncMock(content="Streamed content"))]
                ),
                "break here",
            ]
            for item in completions:
                if item == "break here":
                    raise ValueError("broken")
                yield item

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion, patch(
            "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
        ) as mock_watch:
            watcher = Mock()
            mock_watch.return_value.__enter__.return_value = watcher

            mock_acompletion.side_effect = AsyncMock(side_effect=mock_stream)

            messages = [Message(content="Test message", role="user")]
            response = await lite_llm_chat_model.generate(
                messages=messages, stream=True
            )

            watcher.finish.assert_not_called()

            with pytest.raises(ValueError):
                _ = [item async for item in response]

            mock_watch.assert_called_once_with(stream=True)
            watcher.register_error.assert_called_once()
            watcher.finish.assert_called_once()
