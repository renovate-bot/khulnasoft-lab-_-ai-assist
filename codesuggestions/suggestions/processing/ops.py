from enum import Enum
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "LanguageId",
    "prepend_lang_id",
    "remove_incomplete_lines",
    "trim_by_max_len",
    "trim_by_sep",
    "find_alnum_point",
    "find_position",
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
    SWIFT = 14


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
    LanguageId.SWIFT: {"swift"},
}

_EXTENSION_TO_LANG_ID = {
    value: key for key, values in _ALL_LANGS.items() for value in values
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


def find_alnum_point(value: str, start_index: int = 0) -> tuple[int, int]:
    row = 0
    col = 0

    found_row = -1
    found_col = -1

    for idx, c in enumerate(value):
        if c == "\n":
            # increase the row counter and reset the column one
            row += 1
            col = 0
            continue

        if idx >= start_index and c.isalnum():
            found_row = row
            found_col = col
            break

        col += 1

    return found_row, found_col


def find_position(value: str, point: tuple[int, int]) -> int:
    row = 0
    col = 0

    for pos, c in enumerate(value):
        if (row, col) == point:
            return pos

        if c == "\n":
            row += 1
            col = 0
            continue

        col += 1

    # Check the last position, which is the end of the string
    if (row, col) == point:
        return len(value)

    return -1
