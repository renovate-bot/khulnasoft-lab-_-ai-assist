from enum import Enum
from typing import Optional
from pathlib import Path

__all__ = [
    "LanguageId",
    "LanguageResolver",
    "ModelPromptBuilder",
    "remove_incomplete_lines",
]


class LanguageId(Enum):
    C = 1
    CPP = 2
    CSHARP = 3
    GO = 4
    JAVA = 5
    JS = 6
    PHP = 7
    PYTHON = 8
    RUBY = 9
    RUST = 10
    SCALA = 11
    TS = 12
    KOTLIN = 13


class LanguageResolver:
    ALL_LANGS = {
        LanguageId.C: {"c", "h"},
        LanguageId.CPP: {"cpp", "hpp", "c++", "h++", "cc", "hh", "C", "H"},
        LanguageId.CSHARP: {"cs"},
        LanguageId.GO: {"go"},
        LanguageId.JAVA: {"java"},
        LanguageId.JS: {"js"},
        LanguageId.PHP: {"php", "php3", "php4", "php5", "phps", "phpt"},
        LanguageId.PYTHON: {"py"},
        LanguageId.RUBY: {"rb"},
        LanguageId.RUST: {"rs"},
        LanguageId.SCALA: {"scala"},
        LanguageId.TS: {"ts", "tsx"},
        LanguageId.KOTLIN: {"kts", "kt"},
    }

    EXTENSION_TO_LANG_ID = {
        value: key
        for key, values in ALL_LANGS.items()
        for value in values
    }

    @staticmethod
    def resolve(file_name: str) -> Optional[LanguageId]:
        ext = Path(file_name).suffix.replace(".", "")
        return LanguageResolver.EXTENSION_TO_LANG_ID.get(ext, None)


class ModelPromptBuilder:
    def __init__(self, prompt: str):
        self._prompt = prompt

    def prepend_lang_id(self, lang_id: Optional[LanguageId]):
        if lang_id:
            lang = lang_id.name.lower()
            self._prompt = f"<{lang}>{self.prompt}"

        return self

    @property
    def prompt(self) -> str:
        return self._prompt


def remove_incomplete_lines(s: str, sep: str = "\n") -> str:
    if (index := s.rfind(sep)) != -1:
        return s[:index + 1]

    return s
