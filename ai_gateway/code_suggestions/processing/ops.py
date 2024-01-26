import re
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Union

import numpy as np
from Levenshtein import ratio as levenshtein_ratio
from tree_sitter import Node

from ai_gateway.code_suggestions.processing.typing import LanguageId

__all__ = [
    "prepend_lang_id",
    "remove_incomplete_lines",
    "remove_incomplete_block",
    "trim_by_max_len",
    "trim_by_sep",
    "find_non_whitespace_point",
    "find_cursor_position",
    "find_newline_position",
    "compare_exact",
    "compare_levenshtein",
    "find_common_lines",
    "strip_whitespaces",
]


class _LanguageDef(NamedTuple):
    lang_id: LanguageId
    grammar_name: str
    human_name: str
    extensions: frozenset[str]
    editor_names: frozenset[str]


_ALL_LANGS = {
    _LanguageDef(LanguageId.C, "c", "C", frozenset({"c", "h"}), frozenset({"c"})),
    _LanguageDef(
        LanguageId.CPP,
        "cpp",
        "C++",
        frozenset({"cpp", "hpp", "c++", "h++", "cc", "hh", "C", "H"}),
        frozenset({"cpp"}),
    ),
    _LanguageDef(
        LanguageId.CSHARP, "c_sharp", "C#", frozenset({"cs"}), frozenset({"csharp"})
    ),
    _LanguageDef(LanguageId.GO, "go", "Go", frozenset({"go"}), frozenset({"go"})),
    _LanguageDef(
        LanguageId.JAVA,
        "java",
        "Java",
        frozenset({"java"}),
        frozenset({"java"}),
    ),
    _LanguageDef(
        LanguageId.JS,
        "javascript",
        "JavaScript",
        frozenset({"js", "jsx"}),
        frozenset({"javascript", "javascriptreact"}),
    ),
    _LanguageDef(
        LanguageId.PHP,
        "php",
        "PHP",
        frozenset({"php", "php3", "php4", "php5", "phps", "phpt"}),
        frozenset({"php"}),
    ),
    _LanguageDef(
        LanguageId.PYTHON, "python", "Python", frozenset({"py"}), frozenset({"python"})
    ),
    _LanguageDef(
        LanguageId.RUBY, "ruby", "Ruby", frozenset({"rb"}), frozenset({"ruby"})
    ),
    _LanguageDef(
        LanguageId.RUST, "rust", "Rust", frozenset({"rs"}), frozenset({"rust"})
    ),
    _LanguageDef(
        LanguageId.SCALA, "scala", "Scala", frozenset({"scala"}), frozenset({"scala"})
    ),
    _LanguageDef(
        LanguageId.TS,
        "typescript",
        "TypeScript",
        frozenset({"ts", "tsx"}),
        frozenset({"typescript", "typescriptreact"}),
    ),
    _LanguageDef(
        LanguageId.KOTLIN,
        "kotlin",
        "Kotlin",
        frozenset({"kts", "kt"}),
        frozenset({"kotlin"}),
    ),
}

_LANG_ID_TO_LANG_DEF = {value.lang_id: value for value in _ALL_LANGS}

_EXTENSION_TO_LANG_ID = {
    ext: language.lang_id for language in _ALL_LANGS for ext in language.extensions
}

_EDITOR_LANG_TO_LANG_ID = {
    name: language.lang_id for language in _ALL_LANGS for name in language.editor_names
}

# A new line with a non-indented letter or comment (/*, #, //)
_END_OF_CODE_BLOCK_REGEX = re.compile(r"\n([a-zA-Z]|(\/\*)|(#)|(\/\/))")

# The maximum percentage of the text that can be trimmed to remove an incomplete code block
_MAX_CODE_BLOCK_TRIM_PERCENT = 0.1

_MIN_LEVENSHTEIN_SIMILARITY = 0.9


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


def remove_incomplete_block(
    s: str, max_trim_percent: float = _MAX_CODE_BLOCK_TRIM_PERCENT
) -> str:
    end_of_block = _END_OF_CODE_BLOCK_REGEX.search(
        s, endpos=int(len(s) * max_trim_percent)
    )
    if end_of_block:
        index = end_of_block.start()
        return s[index + 1 :]

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


def lang_from_editor_lang(editor_lang: str) -> Optional[LanguageId]:
    return _EDITOR_LANG_TO_LANG_ID.get(editor_lang, None)


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


def compare_exact(a: str, b: str) -> bool:
    return a == b


def compare_levenshtein(
    a: str, b: str, min_similarity: float = _MIN_LEVENSHTEIN_SIMILARITY
) -> bool:
    """
    Calculates the similarity between two strings and returns whether
    the value is greater than or equal to `min_similarity`.
    The ratio is a normalized levenshtein similarity in the range [1, 0]
    calculated as 1 - normalized_distance.

    Example:
    ----------
    >>> a = "The name of this file is test.js"
    >>> b = "The name of this file is test-1.js"
    >>> compare_levenshtein(a, b, min_similarity=0.95)
        True
    >>> compare_levenshtein(a, b, min_similarity=0.85)
        False

    :param a: A string
    :param b: A string
    :param min_similarity: The value to be compared to the similarity ratio
    :return: Whether the similarity ratio between the two strings is greater than `min_similarity`
    """
    return levenshtein_ratio(a, b) >= min_similarity


def find_common_lines(
    source: list[str],
    target: list[str],
    comparison_func: Callable[[str, str], bool] = compare_exact,
) -> list[tuple]:
    """
    Finds the common strings between two lists, keeping track of repeated ranges.
    Example:
    ----------
    >>> source = ["abc", "def", "g"]
    >>> target = ["abc", "def", "c", "abc"]
    >>> find_common_lines(source, target)
        [(0,1), (3,)]

    The return indicates that target[0] and target[1] match some lines in source and are
    consecutive, target[3] matches but is not consecutive.

    Method:
    ----------
    1. Construct a matrix of size len(source) x len(target) to store the longest common
       subsequence (LCS) lengths.
    2. Fill this matrix by iterating through source and target and checking if the strings at each
       index match using the `comparison_func`.
    3. If they match, update the LCS length by taking the diagonal value and adding 1.
    4. After the matrix is filled, take the max value along each column to find the matching
       indices in target.
    5. Filter out 0s and collect the actual indices in target that match source.
    6. To group consecutive matches, computes the diff between indices and splits them into groups
       wherever the diff is not 1 (i.e. not consecutive).

    :param source: A list of strings to which we compare the target
    :param target: A list of strings we compare against the source
    :param comparison_func: The function used to compare two strings. Defaults to an exact match.

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
            elif comparison_func(source[i - 1], target[j - 1]):
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


async def strip_whitespaces(text: str) -> str:
    return "" if text.isspace() else text
