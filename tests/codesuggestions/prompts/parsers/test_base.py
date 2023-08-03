import pytest

from codesuggestions.prompts.parsers import CodeParser
from codesuggestions.suggestions.processing.base import LanguageId


def test_non_utf8():
    value = b"\xc3\x28"  # Invalid UTF-8 byte sequence

    with pytest.raises(ValueError):
        CodeParser.from_language_id(value, LanguageId.JS)
