from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_gateway.models.agent_model import AgentModel
from ai_gateway.models.base_text import TextGenModelOutput
from ai_gateway.prompts.base import Prompt


class TestAgentModel:
    @pytest.fixture
    def prompt(self) -> Prompt:
        async def _stream_prompt(*_args, **_kwargs):
            for action in [AsyncMock(content="part 1"), AsyncMock(content="part 2")]:
                yield action

        prompt = MagicMock(spec=Prompt)
        prompt.name = "test"
        prompt.ainvoke = AsyncMock(
            side_effect=lambda *_args, **_kwargs: AsyncMock(content="whole part")
        )
        prompt.astream = MagicMock(side_effect=_stream_prompt)

        return prompt

    @pytest.fixture
    def model(self, prompt: Prompt):
        return AgentModel(prompt)

    @pytest.fixture
    def mixtral_prompt(self) -> Prompt:
        async def _stream_prompt(*_args, **_kwargs):
            for action in [
                AsyncMock(content="part\\_1"),
                AsyncMock(content="part\\_2"),
            ]:
                yield action

        prompt = MagicMock(spec=Prompt)
        prompt.name = "test"
        prompt.model_name = "mixtral"
        prompt.ainvoke = AsyncMock(
            side_effect=lambda *_args, **_kwargs: AsyncMock(
                content="underscore\\_mistral output"
            )
        )
        prompt.astream = MagicMock(side_effect=_stream_prompt)

        return prompt

    @pytest.fixture
    def mixtral_model(self, mixtral_prompt: Prompt):
        return AgentModel(mixtral_prompt)

    @pytest.fixture
    def params(self):
        return {"prefix": "def binary_search(s):", "suffix": "end"}

    def test_init(self, prompt, model):
        assert model.prompt == prompt
        assert model.metadata.name == "test"
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

    @pytest.mark.asyncio
    async def test_generate_stream_with_bad_underscore(
        self, mixtral_prompt, mixtral_model, params
    ):
        response = await mixtral_model.generate(params, stream=True)

        content = []
        async for chunk in response:
            content.append(chunk.text)
        assert content == ["part_1", "part_2"]

        mixtral_prompt.astream.assert_called_with(params)

    @pytest.mark.asyncio
    async def test_generate_with_bad_underscore(
        self, mixtral_prompt, mixtral_model, params
    ):
        response = await mixtral_model.generate(params, stream=False)

        assert isinstance(response, TextGenModelOutput)
        assert response.text == "underscore_mistral output"

        mixtral_prompt.ainvoke.assert_called_with(params)
