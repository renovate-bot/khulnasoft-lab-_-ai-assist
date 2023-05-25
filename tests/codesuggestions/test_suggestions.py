import pytest
from unittest.mock import Mock

from codesuggestions.suggestions.base import (
    CodeSuggestionsUseCase,
    CodeSuggestionsUseCaseV2,
    DEFAULT_REPLACEMENT_EMAIL,
    DEFAULT_REPLACEMENT_IPV4,
    DEFAULT_REPLACEMENT_IPV6,
    DEFAULT_REPLACEMENT_SECRET,
)
from codesuggestions.models import BaseModel


@pytest.mark.parametrize(
    "test_content,test_file_name,expected_prompt",
    [
        ("model content", "f.unknown", "model content"),
        ("model content", "f.py", "<python>model content"),
        ("model content", "f.min.js", "<js>model content"),
    ],
)
class TestPromptEngine:
    @staticmethod
    def _model_call(prompt: str) -> str:
        # we return the prompt back to check if it was constructed correctly
        return prompt

    def test_code_suggestions_v2(self, test_content, test_file_name, expected_prompt):
        model = Mock(spec=BaseModel)
        model.side_effect = self._model_call
        u = CodeSuggestionsUseCaseV2(model)

        # prompt returned back by the dummy model
        constructed_prompt = u(test_content, test_file_name)

        assert constructed_prompt == expected_prompt
