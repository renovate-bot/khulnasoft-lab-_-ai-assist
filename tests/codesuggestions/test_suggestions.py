import pytest
from unittest.mock import Mock

from codesuggestions.suggestions.base import (
    CodeSuggestionsUseCaseV2,
)
from codesuggestions.models import TextGenBaseModel, TextGenModelOutput


@pytest.mark.parametrize(
    "test_content,test_file_name,test_third_party,expected_prompt",
    [
        ("model content", "f.unknown", False, "model content"),
        ("model content", "f.py", False, "<python>model content"),
        ("model content", "f.min.js", False, "<js>model content"),
        ("model content", "f.unknown", True, "model content"),
        ("model content", "f.py", True, "<python>model content"),
        ("model content", "f.min.js", True, "<js>model content"),
    ],
)
class TestPromptEngine:
    @staticmethod
    def _model_call(prompt: str) -> TextGenModelOutput:
        # we return the prompt back to check if it was constructed correctly
        return TextGenModelOutput(
            text=prompt,
        )

    def test_code_suggestions_v2(self, test_content, test_file_name, test_third_party, expected_prompt):
        codegen_model = Mock(spec=TextGenBaseModel)
        palm_model = Mock(spec=TextGenBaseModel)

        for model in [codegen_model, palm_model]:
            model.MAX_MODEL_LEN = 1_000  # required in preprocessing
            model.generate = Mock(side_effect=self._model_call)

        u = CodeSuggestionsUseCaseV2(codegen_model, palm_model)

        # prompt returned back by the dummy model
        constructed_prompt = u(test_content, test_file_name, third_party=test_third_party)

        assert constructed_prompt == expected_prompt

        if test_third_party:
            palm_model.generate.assert_called()
        else:
            codegen_model.generate.assert_called()
