import os
from typing import Type
from unittest import mock

import pytest
from anthropic import APITimeoutError, AsyncAnthropic
from gitlab_cloud_connector import GitLabUnitPrimitive
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from litellm.exceptions import Timeout
from pydantic import HttpUrl

from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.prompts.base import Prompt, model_metadata_to_params
from ai_gateway.prompts.config.base import PromptParams
from ai_gateway.prompts.typing import Model, ModelMetadata


class TestPrompt:
    def test_initialize(
        self, prompt: Prompt, unit_primitives: list[GitLabUnitPrimitive]
    ):
        assert prompt.name == "test_prompt"
        assert prompt.unit_primitives == unit_primitives
        assert isinstance(prompt.bound, Runnable)

    def test_build_prompt_template(self, prompt_template):
        prompt_template = Prompt._build_prompt_template(prompt_template)

        assert prompt_template == ChatPromptTemplate.from_messages(
            [("system", "Hi, I'm {{name}}"), ("user", "{{content}}")],
            template_format="jinja2",
        )

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
        response = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

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

        async for c in prompt.astream({"name": "Duo", "content": "What's up?"}):
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
            (
                ChatLiteLLM(
                    model="claude-3-5-sonnet-v2@20241022", custom_llm_provider="vertex_ai"  # type: ignore[call-arg]
                ),
                Timeout,
            ),
        ],
    )
    async def test_timeout(
        self, prompt: Prompt, model: Model, expected_exception: Type
    ):
        with pytest.raises(expected_exception):
            await prompt.ainvoke(
                {"name": "Duo", "content": "Print pi with 400 decimals"}
            )


class TestModelMetadataToParams:
    def test_without_identifier(self):
        model_metadata = ModelMetadata(
            name="model_family",
            provider="provider",
            endpoint=HttpUrl("https://api.example.com"),
            api_key="abcde",
            identifier=None,
        )

        params = model_metadata_to_params(model_metadata)

        assert params == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model_family",
            "custom_llm_provider": "provider",
        }

    def test_with_identifier_no_provider(self):
        model_metadata = ModelMetadata(
            name="model_family",
            provider="provider",
            endpoint=HttpUrl("https://api.example.com"),
            api_key="abcde",
            identifier="model_identifier",
        )

        params = model_metadata_to_params(model_metadata)

        assert params == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model_identifier",
            "custom_llm_provider": "custom_openai",
        }

    def test_with_identifier_with_provider(self):
        model_metadata = ModelMetadata(
            name="model_family",
            provider="provider",
            endpoint=HttpUrl("https://api.example.com"),
            api_key="abcde",
            identifier="custom_provider/model/identifier",
        )

        params = model_metadata_to_params(model_metadata)

        assert params == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model/identifier",
            "custom_llm_provider": "custom_provider",
        }

    def test_with_identifier_with_bedrock_provider(self):
        model_metadata = ModelMetadata(
            name="model_family",
            provider="provider",
            endpoint=HttpUrl("https://api.example.com"),
            api_key="abcde",
            identifier="bedrock/model/identifier",
        )

        params = model_metadata_to_params(model_metadata)

        assert params == {
            "model": "model/identifier",
            "api_key": "abcde",
            "custom_llm_provider": "bedrock",
        }
