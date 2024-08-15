import os
from typing import Type
from unittest import mock

import pytest
from anthropic import APITimeoutError, AsyncAnthropic
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from litellm.exceptions import Timeout
from pydantic import HttpUrl

from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.prompts.base import Prompt
from ai_gateway.prompts.config.base import PromptParams
from ai_gateway.prompts.typing import STUBBED_API_KEY, ModelMetadata


class TestPrompt:
    def test_initialize(
        self, prompt: Prompt, unit_primitives: list[GitLabUnitPrimitive]
    ):
        assert prompt.name == "test_prompt"
        assert prompt.unit_primitives == unit_primitives
        assert isinstance(prompt.bound, Runnable)

    def test_build_messages(self, prompt_template):
        messages = Prompt.build_messages(
            prompt_template, {"name": "Duo", "content": "What's up?"}
        )

        assert messages == [("system", "Hi, I'm Duo"), ("user", "What's up?")]

    def test_instrumentator(self, model_engine: str, model_name: str, prompt: Prompt):
        assert prompt.instrumentator.labels == {
            "model_engine": model_engine,
            "model_name": model_name,
        }

    @pytest.mark.asyncio
    @mock.patch(
        "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
    )
    async def test_ainvoke(
        self, mock_watch: mock.Mock, prompt: Prompt, model_response: str
    ):
        response = await prompt.ainvoke({})

        assert response.content == model_response

        mock_watch.assert_called_with(stream=False)

    @pytest.mark.asyncio
    @mock.patch(
        "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
    )
    async def test_astream(
        self, mock_watch: mock.Mock, prompt: Prompt, model_response: str
    ):
        mock_watcher = mock.AsyncMock()
        mock_watch.return_value.__enter__.return_value = mock_watcher
        response = ""

        async for c in prompt.astream({}):
            response += c.content

            mock_watcher.afinish.assert_not_awaited()  # Make sure we don't finish prematurely

        assert response == model_response

        mock_watch.assert_called_with(stream=True)

        mock_watcher.afinish.assert_awaited_once()


@pytest.mark.skipif(
    # pylint: disable=direct-environment-variable-reference
    os.getenv("REAL_AI_REQUEST") is None,
    # pylint: enable=direct-environment-variable-reference
    reason="3rd party requests not enabled",
)
class TestPromptTimeout:
    @pytest.fixture
    def prompt_options(self):
        yield {"name": "Duo", "content": "Print pi with 400 decimals"}

    @pytest.fixture
    def prompt_params(self):
        yield PromptParams(timeout=0.1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model", "expected_exception"),
        [
            (
                ChatAnthropic(
                    async_client=AsyncAnthropic(), model="claude-3-sonnet-20240229"  # type: ignore[call-arg]
                ),
                APITimeoutError,
            ),
            (
                ChatLiteLLM(
                    model="claude-3-sonnet@20240229", custom_llm_provider="vertex_ai"  # type: ignore[call-arg]
                ),
                Timeout,
            ),
        ],
    )
    async def test_timeout(
        self, prompt: Prompt, model: BaseChatModel, expected_exception: Type
    ):
        with pytest.raises(expected_exception):
            await prompt.ainvoke({})


class TestModelMetadata:
    def test_stubbing_empty_api_key(self):
        params = {
            "endpoint": HttpUrl("http://example.com"),
            "name": "mistral",
            "provider": "litellm",
        }

        metadata = ModelMetadata(**params)
        assert metadata.api_key == STUBBED_API_KEY

        metadata = ModelMetadata(**params, api_key="")
        assert metadata.api_key == STUBBED_API_KEY
