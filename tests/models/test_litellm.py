from typing import AsyncIterator, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_gateway.models import KindLiteLlmModel, LiteLlmChatModel
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.base_chat import Message, Role
from ai_gateway.models.base_text import TextGenModelChunk, TextGenModelOutput
from ai_gateway.models.litellm import LiteLlmTextGenModel


class TestKindLiteLlmModel:
    def test_chat_model(self):
        assert KindLiteLlmModel.MISTRAL.chat_model() == "openai/mistral"
        assert KindLiteLlmModel.MIXTRAL.chat_model() == "openai/mixtral"
        assert KindLiteLlmModel.CODE_GEMMA.chat_model() == "openai/codegemma"
        assert KindLiteLlmModel.CODESTRAL.chat_model() == "openai/codestral"
        assert (
            KindLiteLlmModel.CODESTRAL.chat_model(provider=KindModelProvider.MISTRALAI)
            == "codestral/codestral"
        )

    def test_text_model(self):
        assert (
            KindLiteLlmModel.CODE_GEMMA.text_model()
            == "text-completion-openai/codegemma"
        )
        assert (
            KindLiteLlmModel.CODESTRAL.text_model()
            == "text-completion-openai/codestral"
        )
        assert (
            KindLiteLlmModel.CODESTRAL.text_model(provider=KindModelProvider.MISTRALAI)
            == "text-completion-codestral/codestral"
        )


class TestLiteLlmChatMode:
    @pytest.fixture
    def endpoint(self):
        return "http://127.0.0.1:1111/v1"

    @pytest.fixture
    def api_key(self):
        return "specified-api-key"

    @pytest.fixture
    def lite_llm_chat_model(self, endpoint, api_key):
        return LiteLlmChatModel.from_model_name(
            name="mistral",
            endpoint=endpoint,
            api_key=api_key,
            custom_models_enabled=True,
        )

    @pytest.mark.parametrize(
        (
            "model_name",
            "api_key",
            "provider",
            "custom_models_enabled",
            "provider_keys",
            "expected_name",
            "expected_api_key",
            "expected_engine",
        ),
        [
            (
                "mistral",
                "",
                KindModelProvider.LITELLM,
                True,
                {},
                "openai/mistral",
                "stubbed-api-key",
                "litellm",
            ),
            (
                "mixtral",
                None,
                KindModelProvider.LITELLM,
                True,
                {},
                "openai/mixtral",
                "stubbed-api-key",
                "litellm",
            ),
            (
                "codestral",
                "",
                KindModelProvider.MISTRALAI,
                True,
                {},
                "codestral/codestral",
                "stubbed-api-key",
                "codestral",
            ),
            (
                "codestral",
                None,
                KindModelProvider.MISTRALAI,
                True,
                {"mistral_api_key": "stubbed-mistral-api-key"},
                "codestral/codestral",
                "stubbed-mistral-api-key",
                "codestral",
            ),
        ],
    )
    def test_from_model_name(
        self,
        model_name: str,
        api_key: Optional[str],
        provider: KindModelProvider,
        custom_models_enabled: bool,
        provider_keys: dict,
        expected_name: str,
        expected_api_key: str,
        expected_engine: str,
        endpoint,
    ):
        model = LiteLlmChatModel.from_model_name(
            name=model_name,
            api_key=api_key,
            endpoint=endpoint,
            custom_models_enabled=custom_models_enabled,
            provider=provider,
            provider_keys=provider_keys,
        )

        assert model.metadata.name == expected_name
        assert model.endpoint == endpoint
        assert model.api_key == expected_api_key
        assert model.metadata.engine == expected_engine

        model = LiteLlmChatModel.from_model_name(name=model_name, api_key=None)

        assert model.endpoint is None
        assert model.api_key == "stubbed-api-key"

        if provider == KindModelProvider.LITELLM:
            with pytest.raises(ValueError) as exc:
                LiteLlmChatModel.from_model_name(name=model_name, endpoint=endpoint)
            assert str(exc.value) == "specifying custom models endpoint is disabled"

            with pytest.raises(ValueError) as exc:
                LiteLlmChatModel.from_model_name(name=model_name, api_key="api-key")
            assert str(exc.value) == "specifying custom models endpoint is disabled"

    @pytest.mark.asyncio
    async def test_generate(self, lite_llm_chat_model, endpoint, api_key):
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
                api_key=api_key,
                api_base=endpoint,
                timeout=30.0,
                stop=["</new_code>"],
            )

    @pytest.mark.asyncio
    async def test_generate_stream(self, lite_llm_chat_model, endpoint, api_key):
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
                api_key=api_key,
                api_base=endpoint,
                timeout=30.0,
                stop=["</new_code>"],
            )

            mock_watch.assert_called_once_with(stream=True)
            watcher.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_instrumented(self, lite_llm_chat_model):
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


class TestLiteLlmTextGenModel:
    @pytest.fixture
    def endpoint(self):
        return "http://127.0.0.1:4000"

    @pytest.fixture
    def api_key(self):
        return "specified-api-key"

    @pytest.fixture
    def lite_llm_text_model(self, endpoint, api_key):
        return LiteLlmTextGenModel.from_model_name(
            name="codegemma",
            endpoint=endpoint,
            api_key=api_key,
            custom_models_enabled=True,
        )

    @pytest.mark.parametrize(
        (
            "model_name",
            "api_key",
            "provider",
            "custom_models_enabled",
            "provider_keys",
            "expected_name",
            "expected_api_key",
            "expected_engine",
        ),
        [
            (
                "codegemma",
                "",
                KindModelProvider.LITELLM,
                True,
                {},
                "text-completion-openai/codegemma",
                "stubbed-api-key",
                "litellm",
            ),
            (
                "codegemma",
                None,
                KindModelProvider.LITELLM,
                True,
                {},
                "text-completion-openai/codegemma",
                "stubbed-api-key",
                "litellm",
            ),
            (
                "codestral",
                None,
                KindModelProvider.MISTRALAI,
                True,
                {},
                "text-completion-codestral/codestral",
                "stubbed-api-key",
                "codestral",
            ),
            (
                "codestral",
                "",
                KindModelProvider.MISTRALAI,
                True,
                {"mistral_api_key": "stubbed-mistral-api-key"},
                "text-completion-codestral/codestral",
                "stubbed-mistral-api-key",
                "codestral",
            ),
        ],
    )
    def test_from_model_name(
        self,
        model_name: str,
        api_key: Optional[str],
        provider: KindModelProvider,
        custom_models_enabled: bool,
        provider_keys: dict,
        expected_name: str,
        expected_api_key: str,
        expected_engine: str,
        endpoint,
    ):
        model = LiteLlmTextGenModel.from_model_name(
            name=model_name,
            api_key=api_key,
            endpoint=endpoint,
            custom_models_enabled=custom_models_enabled,
            provider=provider,
            provider_keys=provider_keys,
        )

        assert model.metadata.name == expected_name
        assert model.endpoint == endpoint
        assert model.api_key == expected_api_key
        assert model.metadata.engine == expected_engine

        model = LiteLlmTextGenModel.from_model_name(name=model_name, api_key=None)

        assert model.endpoint is None
        assert model.api_key == "stubbed-api-key"

        if provider == KindModelProvider.LITELLM:
            with pytest.raises(ValueError) as exc:
                LiteLlmTextGenModel.from_model_name(name=model_name, endpoint=endpoint)
            assert str(exc.value) == "specifying custom models endpoint is disabled"

            with pytest.raises(ValueError) as exc:
                LiteLlmTextGenModel.from_model_name(name=model_name, api_key="api-key")
            assert str(exc.value) == "specifying custom models endpoint is disabled"

    @pytest.mark.asyncio
    async def test_generate(self, lite_llm_text_model, endpoint, api_key):

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
            mock_acompletion.return_value = AsyncMock(
                choices=[AsyncMock(message=AsyncMock(content="Test response"))]
            )
            _generate_args = {
                "stream": False,
                "temperature": 0.9,
                "max_output_tokens": 10,
                "top_p": 0.95,
                "top_k": 0,
            }
            output = await lite_llm_text_model.generate(
                prefix="def hello_world():", **_generate_args
            )
            assert isinstance(output, TextGenModelOutput)
            assert output.text == "Test response"

    @pytest.mark.asyncio
    async def test_generate_stream(self, lite_llm_text_model, endpoint, api_key):
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

            response = await lite_llm_text_model.generate(
                prefix="Test message",
                stream=True,
            )

            content = []
            async for chunk in response:
                content.append(chunk.text)
            assert content == ["Streamed content"]

            mock_watch.assert_called_once_with(stream=True)
            watcher.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_instrumented(self, lite_llm_text_model):
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

            response = await lite_llm_text_model.generate(
                prefix="Test message", stream=True
            )

            watcher.finish.assert_not_called()

            with pytest.raises(ValueError):
                _ = [item async for item in response]

            mock_watch.assert_called_once_with(stream=True)
            watcher.register_error.assert_called_once()
            watcher.finish.assert_called_once()
