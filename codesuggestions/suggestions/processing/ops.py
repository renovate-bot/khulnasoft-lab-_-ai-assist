from enum import Enum
from pathlib import Path
from typing import Optional, Union

from tree_sitter import Node

__all__ = [
    "LanguageId",
    "prepend_lang_id",
    "remove_incomplete_lines",
    "trim_by_max_len",
    "trim_by_sep",
    "find_alnum_point",
    "find_cursor_position",
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


def split_on_point(
    source_code: str, point: tuple[int, int]
) -> tuple[Optional[str], Optional[str]]:
    """
    Splits the source_code into a prefix and a suffix.
    Returns (None,None) if the splitting point is invalid.
    """
    pos = find_cursor_position(source_code, point)
    if pos == -1:
        return (None, None)

    prefix = source_code[:pos]
    suffix = source_code[pos:]
    return (prefix, suffix)


def find_cursor_position(source_code: str, point: tuple[int, int]) -> int:
    """
    Converts a 2D point to its 1D position in the source_code.
    """
    if not source_code:
        return -1

    row, col = point
    lines = source_code.splitlines()

    if row >= len(lines) or col > len(lines[row]):
        return -1

    pos = 0
    for line in lines[:row]:
        pos += len(line) + 1
    pos += col
    return pos


def convert_point_to_relative_point_in_node(
    node: Node, point: tuple[int, int]
) -> tuple[int, int]:
    """
    Converts the global point to the relative point within the node.
    """
    row = point[0] - node.start_point[0]
    col = point[1] - node.start_point[1]
    return (row, col)
