import pytest
from unittest.mock import Mock

from codesuggestions.suggestions.base import (
    CodeSuggestionsUseCaseV2,
)
from codesuggestions.suggestions.processing import ModelEngineBase


@pytest.mark.parametrize(
    "content,file_name,third_party,expected_completion",
    [
        ("codegen model content", "f.unknown", False, "codegen model content"),
        ("palm model content", "f.unknown", True, "palm model content"),
    ],
)
class TestCodeSuggestionUseCases:
    @staticmethod
    def _engine_call(prompt: str, _: str) -> str:
        # return the prompt back
        return prompt

    def test_code_suggestions_v2(self, content, file_name, third_party, expected_completion):
        codegen_engine = Mock(spec=ModelEngineBase)
        palm_engine = Mock(spec=ModelEngineBase)

        codegen_engine.generate_completion = Mock(side_effect=self._engine_call)
        palm_engine.generate_completion = Mock(side_effect=self._engine_call)

        u = CodeSuggestionsUseCaseV2(codegen_engine, palm_engine)

        completion = u(content, file_name, third_party=third_party)

        assert completion == expected_completion

        if third_party:
            palm_engine.generate_completion.assert_called()
        else:
            codegen_engine.generate_completion.assert_called()
