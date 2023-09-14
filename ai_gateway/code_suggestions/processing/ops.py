from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np
from transformers import PreTrainedTokenizer
from tree_sitter import Node

from ai_gateway.code_suggestions.processing.typing import CodeContent, LanguageId

__all__ = [
    "prepend_lang_id",
    "remove_incomplete_lines",
    "trim_by_max_len",
    "trim_by_sep",
    "find_non_whitespace_point",
    "find_cursor_position",
    "truncate_content",
    "find_newline_position",
    "find_common_lines",
    "strip_whitespaces",
]


class _LanguageDef(NamedTuple):
    lang_id: LanguageId
    grammar_name: str
    human_name: str
    extensions: frozenset[str]


_ALL_LANGS = {
    _LanguageDef(LanguageId.C, "c", "C", frozenset({"c", "h"})),
    _LanguageDef(
        LanguageId.CPP,
        "cpp",
        "C++",
        frozenset({"cpp", "hpp", "c++", "h++", "cc", "hh", "C", "H"}),
    ),
    _LanguageDef(LanguageId.CSHARP, "c_sharp", "C#", frozenset({"cs"})),
    _LanguageDef(LanguageId.GO, "go", "Go", frozenset({"go"})),
    _LanguageDef(LanguageId.JAVA, "java", "Java", frozenset({"java"})),
    _LanguageDef(LanguageId.JS, "javascript", "JavaScript", frozenset({"js", "jsx"})),
    _LanguageDef(
        LanguageId.PHP,
        "php",
        "PHP",
        frozenset({"php", "php3", "php4", "php5", "phps", "phpt"}),
    ),
    _LanguageDef(LanguageId.PYTHON, "python", "Python", frozenset({"py"})),
    _LanguageDef(LanguageId.RUBY, "ruby", "Ruby", frozenset({"rb"})),
    _LanguageDef(LanguageId.RUST, "rust", "Rust", frozenset({"rs"})),
    _LanguageDef(LanguageId.SCALA, "scala", "Scala", frozenset({"scala"})),
    _LanguageDef(LanguageId.TS, "typescript", "TypeScript", frozenset({"ts", "tsx"})),
    _LanguageDef(LanguageId.KOTLIN, "kotlin", "Kotlin", frozenset({"kts", "kt"})),
}

_LANG_ID_TO_LANG_DEF = {value.lang_id: value for value in _ALL_LANGS}

_EXTENSION_TO_LANG_ID = {
    ext: language.lang_id for language in _ALL_LANGS for ext in language.extensions
}


class ProgramLanguage:
    def __init__(self, lang_id: LanguageId):
        self._lang_id = lang_id
        self._lang_def = _LANG_ID_TO_LANG_DEF.get(lang_id)

    def __getattr__(self, name):
        return getattr(self._lang_def, name)

    @classmethod
    def from_language_id(cls, lang_id: LanguageId):
        return ProgramLanguage(lang_id)


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


def find_non_whitespace_point(value: str, start_index: int = 0) -> tuple[int, int]:
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

        if idx >= start_index and not c.isspace():
            found_row = row
            found_col = col
            break

        col += 1

    return found_row, found_col


def find_newline_position(value: str, start_index: int = 0) -> int:
    """
    Finds the nearest newline position close to `start_index`
    """
    substring = value[:start_index]
    substring_rstrip = substring.rstrip(" \t")

    if substring_rstrip.endswith("\n"):
        return len(substring)

    for idx, c in enumerate(value[start_index:]):
        if c == "\n":
            return len(substring) + idx + 1

    return -1


def find_common_lines(source: list[str], target: list[str]) -> list[tuple]:
    """
    Finds the common strings between two lists, keeping track of repeated ranges.
    Example:
    ----------
    >>> source = ["abc", "def", "g"]
    >>> target = ["abc", "def", "c", "abc"]
    >>> find_common_lines(source, target)
        [(0,1), (3,)]

    :param source: A list of strings to which we compare the target
    :param target: A list of strings we compare against the source
    :return: A list of indices of common strings grouped if they are consecutive lines
    """

    len_source = len(source)
    len_target = len(target)

    # The 0th row and column always contain zero values to simplify
    # the LCS algorithm implementation
    L = np.zeros((len_source + 1, len_target + 1), dtype=int)

    # Tabulated implementation for the LCS problem.
    # Complexity: O(len_source*len_target)
    # Goal: find all common lines and their sequences to collect them into groups later
    for i in range(len_source + 1):
        for j in range(len_target + 1):
            if i == 0 or j == 0:
                L[i, j] = 0
            elif source[i - 1] == target[j - 1]:
                # Optimization: start groups of size larger than `1` with `2`, otherwise start with `1`
                # Goal: when getting the maximum over the rows, we need to take larger groups into account first
                prev_match = L[i - 1, j - 1]
                L[i - 1, j - 1] = prev_match + 1 if prev_match == 1 else prev_match

                # The LCS step according to the tabulated implementation
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = 0

    # Get the line numbers with the max value, the length of the 1D array equals to `len_source+1`
    target_max = L.argmax(axis=0)

    # Collect only those lines that match `source`.
    # Note: since we padded the L matrix with zeros, we need to trim the array when getting the indices
    target_lines = target_max > 0
    target_matches = target_max[target_lines]
    target_lines_idx = np.where(target_lines[1:])[0]

    if len(target_lines_idx) == 0:
        return []

    # Group common lines
    # Groups of size larger than `1` always contain consecutive lines
    # E.g.:
    # Input: [0,4,5,6,7]
    # Output: [(0,), (4,5,6), (7,)]
    diff_matches = np.diff(target_matches)
    groups = np.split(target_lines_idx, np.where(diff_matches != 1)[0] + 1)
    groups = list(map(tuple, groups))

    return groups


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


def truncate_content(
    tokenizer: PreTrainedTokenizer,
    val: str,
    max_length: int,
    truncation_side: str = "left",
) -> CodeContent:
    prev_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side

    tokens = tokenizer(
        val,
        max_length=max_length,
        truncation=True,
        return_attention_mask=False,
        add_special_tokens=False,
    )

    decoded = tokenizer.decode(tokens["input_ids"])
    tokenizer.truncation_side = prev_truncation_side

    return CodeContent(
        text=decoded,
        length_tokens=len(tokens["input_ids"]),
    )


def strip_whitespaces(text: str) -> str:
    return "" if text.isspace() else text
