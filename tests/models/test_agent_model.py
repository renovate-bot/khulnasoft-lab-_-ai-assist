from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.runnables import Runnable

from ai_gateway.models.agent_model import AgentModel
from ai_gateway.models.base_text import TextGenModelOutput


class TestAgentModel:
    @pytest.fixture
    def prompt(self):
        async def _stream_prompt(*_args, **_kwargs):
            for action in [AsyncMock(content="part 1"), AsyncMock(content="part 2")]:
                yield action

        prompt = Mock(spec=Runnable)
        prompt.ainvoke = AsyncMock(
            side_effect=lambda *_args, **_kwargs: AsyncMock(content="whole part")
        )
        prompt.astream = Mock(side_effect=_stream_prompt)

        return prompt

    @pytest.fixture
    def model(self, prompt):
        return AgentModel(prompt)

    @pytest.fixture
    def params(self):
        return {"prefix": "def binary_search(s):", "suffix": "end"}

    def test_init(self, prompt, model):
        assert model.prompt == prompt
        assert model.metadata.name == prompt.name
        assert model.metadata.engine == "agent"

    @pytest.mark.asyncio
    async def test_generate(self, prompt, model, params):
        response = await model.generate(params, stream=False)

        assert isinstance(response, TextGenModelOutput)
        assert response.text == "whole part"

        prompt.ainvoke.assert_called_with(params)

    @pytest.mark.asyncio
    async def test_generate_stream(self, prompt, model, params):
        response = await model.generate(params, stream=True)

        content = []
        async for chunk in response:
            content.append(chunk.text)
        assert content == ["part 1", "part 2"]

        prompt.astream.assert_called_with(params)
