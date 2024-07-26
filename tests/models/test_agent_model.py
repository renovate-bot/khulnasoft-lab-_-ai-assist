from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.runnables import Runnable, chain

from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.models.agent_model import AgentModel
from ai_gateway.models.base_text import TextGenModelOutput


class TestAgentModel:
    @pytest.fixture
    def agent(self):
        async def _stream_agent(*_args, **_kwargs):
            for action in [AsyncMock(content="part 1"), AsyncMock(content="part 2")]:
                yield action

        agent = Mock(spec=Runnable)
        agent.ainvoke = AsyncMock(
            side_effect=lambda *_args, **_kwargs: AsyncMock(content="whole part")
        )
        agent.astream = Mock(side_effect=_stream_agent)

        return agent

    @pytest.fixture
    def model(self, agent):
        return AgentModel(agent)

    @pytest.fixture
    def params(self):
        return {"prefix": "def binary_search(s):", "suffix": "end"}

    def test_init(self, agent, model):
        assert model.agent == agent
        assert model.metadata.name == agent.name
        assert model.metadata.engine == "agent"

    @pytest.mark.asyncio
    async def test_generate(self, agent, model, params):
        response = await model.generate(params, stream=False)

        assert isinstance(response, TextGenModelOutput)
        assert response.text == "whole part"

        agent.ainvoke.assert_called_with(params)

    @pytest.mark.asyncio
    async def test_generate_stream(self, agent, model, params):
        response = await model.generate(params, stream=True)

        content = []
        async for chunk in response:
            content.append(chunk.text)
        assert content == ["part 1", "part 2"]

        agent.astream.assert_called_with(params)
