import pytest
from unittest.mock import Mock

from codesuggestions.suggestions.base import (
    CodeSuggestionsUseCase, DEFAULT_REPLACEMENT_EMAIL, DEFAULT_REPLACEMENT_IPV4, DEFAULT_REPLACEMENT_IPV6,
)
from codesuggestions.models import BaseModel


@pytest.mark.parametrize(
    "test_content,expected_output", [
        ("mask ip 127.0.0.1 address", f"mask ip {DEFAULT_REPLACEMENT_IPV4} address"),
        ("mask ip :: address", f"mask ip {DEFAULT_REPLACEMENT_IPV6} address"),
        ("mask email mail@box.com address", f"mask email {DEFAULT_REPLACEMENT_EMAIL} address"),
        ("mask email mail@box.com and ip 127.0.0.1 and another ip ::255.255.255.255 and date 01.10.02",
         f"mask email {DEFAULT_REPLACEMENT_EMAIL} and ip {DEFAULT_REPLACEMENT_IPV4} "
         f"and another ip {DEFAULT_REPLACEMENT_IPV6} and date 01.10.02")
    ]
)
def test_redact(test_content, expected_output):
    model = Mock(spec=BaseModel, return_value=test_content)
    u = CodeSuggestionsUseCase(model)

    assert u("unused_random_prompt") == expected_output
