import os
from typing import Type

import pytest
from anthropic import APITimeoutError, AsyncAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import HttpUrl

from ai_gateway.agents.base import Agent
from ai_gateway.agents.config.base import AgentParams
from ai_gateway.agents.typing import STUBBED_API_KEY, ModelMetadata
from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic


class TestAgent:
    def test_initialize(self, agent: Agent, unit_primitives: list[GitLabUnitPrimitive]):
        assert agent.name == "test_agent"
        assert agent.unit_primitives == unit_primitives
        assert isinstance(agent.bound, Runnable)

    def test_build_messages(self, prompt_template):
        messages = Agent.build_messages(
            prompt_template, {"name": "Duo", "content": "What's up?"}
        )

        assert messages == [("system", "Hi, I'm Duo"), ("user", "What's up?")]


@pytest.mark.skipif(
    # pylint: disable=direct-environment-variable-reference
    os.getenv("REAL_AI_REQUEST") is None,
    # pylint: enable=direct-environment-variable-reference
    reason="3rd party requests not enabled",
)
class TestAgentTimeout:
    @pytest.fixture
    def agent_options(self):
        yield {"name": "Duo", "content": "Print pi with 400 decimals"}

    @pytest.fixture
    def agent_params(self):
        yield AgentParams(timeout=0.1)

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
        ],
    )
    async def test_timeout(
        self, agent: Agent, model: BaseChatModel, expected_exception: Type
    ):
        with pytest.raises(expected_exception):
            await agent.ainvoke({})


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
