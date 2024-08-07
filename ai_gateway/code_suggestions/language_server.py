from typing import Optional

import structlog
from packaging.version import InvalidVersion, Version

__all__ = ["LanguageServerVersion"]

log = structlog.stdlib.get_logger("codesuggestions")

ADVANCED_CONTEXT_LANGUAGE_SERVER_VERSION = Version("4.21.0")


class LanguageServerVersion:
    @classmethod
    def from_string(cls, version: Optional[str]) -> "LanguageServerVersion":
        version = version or "0.0.0"

        try:
            return LanguageServerVersion(Version(version))
        except InvalidVersion:
            log.warning(
                "Invalid X-Gitlab-Language-Server-Version header passed.",
                version=version,
            )
            return LanguageServerVersion(Version("0.0.0"))

    def __init__(self, version: Version) -> None:
        self.version = version

    def supports_advanced_context(self) -> bool:
        return self.version >= ADVANCED_CONTEXT_LANGUAGE_SERVER_VERSION
