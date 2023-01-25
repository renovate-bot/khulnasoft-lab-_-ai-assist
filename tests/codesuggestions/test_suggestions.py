import pytest
from unittest.mock import Mock

from codesuggestions.suggestions.base import (
    CodeSuggestionsUseCase,
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
        ("generated discord token MTk4NjIyNDgzNDcxOTI1MjQ4.Cl2FMQ.ZnCjm1XVW7vRze4b7Cq4se7kKWs\nin the file",
         f"generated discord token {DEFAULT_REPLACEMENT_SECRET}\nin the file"),
        ("if api_key == 'password':\n\tprint('password')\n",
         f"if api_key == '{DEFAULT_REPLACEMENT_SECRET}':\n\tprint('password')\n"),
        ("aws_secret_access_key: 'key'\napikey_myservice: 'another key'",
         f"aws_secret_access_key: '{DEFAULT_REPLACEMENT_SECRET}'\napikey_myservice: '{DEFAULT_REPLACEMENT_SECRET}'")
    ]
)
def test_redact(test_content, expected_output):
    model = Mock(spec=BaseModel, return_value=test_content)
    u = CodeSuggestionsUseCase(model)

    assert u("unused_random_prompt") == expected_output
