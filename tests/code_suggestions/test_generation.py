from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_gateway.code_suggestions import CodeGenerations
from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.post.generations import PostProcessor
from ai_gateway.code_suggestions.processing.pre import (
    PromptBuilderBase,
    TokenStrategyBase,
)
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import SafetyAttributes, TextGenBaseModel, TextGenModelOutput


class InstrumentorMock(Mock):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.watcher = Mock()

    @contextmanager
    def watch(self, _prompt: str, **_kwargs: Any):
        yield self.watcher


@pytest.mark.asyncio
class TestCodeGeneration:
    @pytest.fixture(scope="class")
    def use_case(self):
        model = Mock(spec=TextGenBaseModel)
        model.MAX_MODEL_LEN = 2048

        use_case = CodeGenerations(model, Mock(spec=TokenStrategyBase))
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)
        use_case.prompt_builder = Mock(spec=PromptBuilderBase)

        yield use_case

    @pytest.mark.parametrize("prompt_version", [(1), (2)])
    async def test_execute_with_prompt_version(
        self, use_case: CodeGenerations, prompt_version
    ):
        use_case.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="output", score=0, safety_attributes=SafetyAttributes()
            )
        )
        with patch.object(PostProcessor, "process") as mock:
            actual = await use_case.execute(
                "prefix",
                "test.py",
                editor_lang=LanguageId.PYTHON,
                raw_prompt="test prompt",
                prompt_version=prompt_version,
            )
            mock.assert_not_called if prompt_version == 2 else mock.assert_called
