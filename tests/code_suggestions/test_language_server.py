import pytest
from packaging.version import Version

from ai_gateway.code_suggestions.language_server import LanguageServerVersion


@pytest.mark.asyncio
class TestLanguageServer:
    @pytest.mark.parametrize(
        ("semver", "expected_version", "supports_advanced_context"),
        [
            (None, Version("0.0.0"), False),
            ("invalid version", Version("0.0.0"), False),
            ("0.0.0", Version("0.0.0"), False),
            ("4.15.0", Version("4.15.0"), False),
            ("4.21.0", Version("4.21.0"), True),
            ("5.0.0", Version("5.0.0"), True),
            ("5.0.0-beta.1", Version("5.0.0-beta.1"), True),
            ("999.99.9", Version("999.99.9"), True),
        ],
    )
    async def test_supports_advanced_context(
        self, semver, expected_version, supports_advanced_context
    ):
        subject = LanguageServerVersion.from_string(semver)
        assert subject.version == expected_version
        assert subject.supports_advanced_context() == supports_advanced_context
