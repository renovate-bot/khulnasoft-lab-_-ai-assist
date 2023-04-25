from enum import Enum
from typing import Optional
from pathlib import Path

__all__ = [
    "LanguageId",
    "LanguageResolver",
    "ModelPromptEncoder",
    "ModelPromptDecoder",
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

    @staticmethod
    def from_file_name(file_name: str) -> Optional[LanguageId]:
        ext = Path(file_name).suffix.replace(".", "")
        for lang, all_ext in LanguageResolver.ALL_LANGS.items():
            if ext in all_ext:
                return lang

        return None


class ModelPromptEncoder:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def prepend_lang_id(self, lang_id: Optional[LanguageId]):
        if lang_id:
            lang = lang_id.name.lower()
            self.prompt = f"<{lang}>{self.prompt}"

        return self

    @property
    def encoded(self) -> str:
        return self.prompt


class ModelPromptDecoder:
    pass
