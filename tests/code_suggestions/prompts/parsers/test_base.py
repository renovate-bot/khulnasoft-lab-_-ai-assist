import pytest

from ai_gateway.code_suggestions.processing.base import LanguageId
from ai_gateway.code_suggestions.prompts.parsers import CodeParser


@pytest.mark.parametrize("lang_id", [None])
@pytest.mark.asyncio
async def test_unsupported_languages(lang_id: LanguageId):
    with pytest.raises(ValueError):
        await CodeParser.from_language_id("import Foundation", lang_id)


@pytest.mark.asyncio
async def test_non_utf8():
    value = b"\xc3\x28"  # Invalid UTF-8 byte sequence

    with pytest.raises(ValueError):
        await CodeParser.from_language_id(value, LanguageId.JS)
