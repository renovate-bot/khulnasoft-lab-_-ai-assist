from enum import Enum
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "LanguageId",
    "prepend_lang_id",
    "remove_incomplete_lines",
    "trim_by_max_len",
    "trim_by_sep",
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


_ALL_LANGS = {
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

_EXTENSION_TO_LANG_ID = {
    value: key
    for key, values in _ALL_LANGS.items()
    for value in values
}


def prepend_lang_id(s: str, lang_id: Optional[LanguageId]):
    if lang_id:
        lang = lang_id.name.lower()
        s = f"<{lang}>{s}"

    return s


def remove_incomplete_lines(s: str, sep: str = "\n") -> str:
    if (index := s.rfind(sep)) > 0:
        return s[:index]

    return s


def trim_by_max_len(s: str, max_context_size: int) -> str:
    if max_context_size < 1:
        raise ValueError("expected `max_context_size` greater or equal to 1")
    return s[-max_context_size:]


def trim_by_sep(s: str, sep: str = "```") -> str:
    if (index := s.find(sep)) != -1:
        return s[:index]

    return s


def lang_from_filename(file_name: Union[str, Path]) -> Optional[LanguageId]:
    ext = Path(file_name).suffix.replace(".", "")
    return _EXTENSION_TO_LANG_ID.get(ext, None)
