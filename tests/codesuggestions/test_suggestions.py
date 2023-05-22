import pytest
from unittest.mock import Mock

from codesuggestions.suggestions.base import (
    CodeSuggestionsUseCase,
    CodeSuggestionsUseCaseV2,
    DEFAULT_REPLACEMENT_EMAIL,
    DEFAULT_REPLACEMENT_IPV4,
    DEFAULT_REPLACEMENT_IPV6,
    DEFAULT_REPLACEMENT_SECRET
)
from codesuggestions.models import BaseModel


@pytest.mark.parametrize(
    "test_content,expected_output", [
        ("mask ip 127.0.0.1 address", f"mask ip {DEFAULT_REPLACEMENT_IPV4} address"),
        ("mask ip :: address", f"mask ip {DEFAULT_REPLACEMENT_IPV6} address"),
        ("mask email mail@box.com address", f"mask email {DEFAULT_REPLACEMENT_EMAIL} address"),
        ("mask email mail@box.com and ip 127.0.0.1 and another ip ::255.255.255.255 and date 01.10.02",
         f"mask email {DEFAULT_REPLACEMENT_EMAIL} and ip {DEFAULT_REPLACEMENT_IPV4} "
         f"and another ip {DEFAULT_REPLACEMENT_IPV6} and date 01.10.02"),
        ("how to clone repo: https://username:encrypted_token@gitlab.com/namespace/project.git",
         f"how to clone repo: https://username:{DEFAULT_REPLACEMENT_SECRET}@gitlab.com/namespace/project.git"),
        ("generated discord token MTk4NjIyNDgzNDcxOTI1MjQ4.Cl2FMQ.ZnCjm1XVW7vRze4b7Cq4se7kKWs\nin the file\n",
         f"generated discord token {DEFAULT_REPLACEMENT_SECRET}\nin the file\n"),
        ("if api_key == 'password':\n\tprint('password')\n",
         f"if api_key == '{DEFAULT_REPLACEMENT_SECRET}':\n\tprint('password')\n"),
        ("aws_secret_access_key: 'key'\napikey_myservice: 'another key'\n",
         f"aws_secret_access_key: '{DEFAULT_REPLACEMENT_SECRET}'\napikey_myservice: '{DEFAULT_REPLACEMENT_SECRET}'\n")
    ]
)
class TestRedactPII:
    def test_code_suggestions_v1(self, test_content, expected_output):
        model = Mock(spec=BaseModel, return_value=test_content)
        u = CodeSuggestionsUseCase(model)

        assert u("unused_random_prompt") == expected_output

    def test_code_suggestions_v2(self, test_content, expected_output):
        model = Mock(spec=BaseModel, return_value=test_content)
        u = CodeSuggestionsUseCaseV2(model)

        assert u("unused_random_prompt", "file.unused") == expected_output


@pytest.mark.parametrize(
    "test_content,test_file_name,expected_prompt", [
        ("model content", "f.unknown", "model content"),
        ("model content", "f.py", "<python>model content"),
        ("model content", "f.min.js", "<js>model content"),
    ]
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
