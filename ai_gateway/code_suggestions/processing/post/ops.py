import re
from collections import Counter
from typing import Any, Optional

import structlog

from ai_gateway.code_suggestions.processing.ops import (
    find_common_lines,
    find_cursor_position,
    find_newline_position,
    find_non_whitespace_point,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId
from ai_gateway.prompts.parsers import CodeParser

__all__ = [
    "clean_model_reflection",
    "trim_by_min_allowed_context",
    "fix_end_block_errors",
    "strip_code_block_markdown",
    "prepend_new_line",
]

log = structlog.stdlib.get_logger("codesuggestions")


_COMMENT_IDENTIFIERS = ["/*", "//", "#"]
_SPECIAL_CHARS = "()[];.,$%&^*@#!{}/"
_RE_MARKDOWN_CODE_BLOCK_BEGIN = re.compile(r"^`{3}\S*\n", flags=re.MULTILINE)


def clean_model_reflection(context: str, completion: str, **kwargs: Any) -> str:
    def _is_single_line_comment(lines: list[str]):
        return len(lines) == 1 and lines[0].lstrip().startswith(
            tuple(_COMMENT_IDENTIFIERS)
        )

    def _with_special_characters(counter: Counter, min_p: float):
        special_characters_count = sum(
            [counter.get(c, 0) for c in tuple(_SPECIAL_CHARS)]
        )
        total_count = sum(counter.values())

        return (special_characters_count / total_count) >= min_p

    def _with_low_diversity(counter: Counter, min_p: float):
        unique_count = len(counter)
        total_count = sum(counter.values())

        return (unique_count / total_count) >= min_p

    def _is_large_group(
        group: tuple,
        lines: list[str],
        min_block_size: int = 5,
        min_special_chars: float = 0.25,
        min_diversity_chars: float = 0.35,
    ):
        counter = Counter("".join(line.strip() for line in lines))
        total_count = sum(counter.values())

        return (
            len(group) >= min_block_size
            and total_count > 0
            and not _with_special_characters(counter, min_special_chars)
            and not _with_low_diversity(counter, min_diversity_chars)
        )

    text = f"{context}{completion}"

    br_pos = find_newline_position(text, start_index=len(context))
    if br_pos == -1:
        # Only the current line was completed, no need to dedupe completion
        return completion

    lines_before = _split_code_lines(text[:br_pos])
    lines_after = _split_code_lines(text[br_pos:])

    common_lines = find_common_lines(
        source=[line.strip() for line in lines_before],
        target=[line.strip() for line in lines_after],
    )

    prev_line = 0
    lines_completion = []
    for group in common_lines:
        start_line, end_line = group[0], group[-1]
        target_lines = lines_after[start_line : end_line + 1]
        lines_completion.extend(lines_after[prev_line:start_line])

        if not (
            _is_single_line_comment(target_lines)
            or _is_large_group(group, target_lines, **kwargs)
        ):
            # Add appropriate lines to the final completion
            # and ignore other lines
            lines_completion.extend(target_lines)

        prev_line = end_line + 1

    # Add remaining lines to the completion list
    lines_completion.extend(lines_after[prev_line:])

    # Get the completion of the current line + processed lines
    completion = text[len(context) : br_pos]
    completion = "".join([completion, *lines_completion])

    return completion


def trim_by_min_allowed_context(
    prefix: str,
    completion: str,
    lang_id: Optional[LanguageId] = None,
) -> str:
    code_sample = f"{prefix}{completion}"
    len_prefix = len(prefix)
    target_point = find_non_whitespace_point(code_sample, start_index=len_prefix)
    if target_point == (-1, -1):
        return completion

    try:
        parser = CodeParser.from_language_id(
            code_sample,
            lang_id,
        )
        context = parser.min_allowed_context(target_point)
        end_pos = find_cursor_position(code_sample, context.end)
        if end_pos == -1:
            return completion

        out = code_sample[len_prefix:end_pos]
    except ValueError as e:
        log.warning(f"Failed to parse code: {e}")
        out = completion

    return out


def fix_end_block_errors(
    prefix: str,
    completion: str,
    suffix: str,
    attempts: int = 4,
    lang_id: Optional[LanguageId] = None,
) -> str:
    total_errors = 0
    completion_lookup = completion

    try:
        for i in range(1, min(attempts, len(completion)) + 2):
            code_sample = f"{prefix}{completion_lookup}{suffix}"
            parser = CodeParser.from_language_id(code_sample, lang_id)

            num_errors = len(parser.errors())

            if num_errors == 0 or num_errors < total_errors:
                return completion_lookup

            total_errors = num_errors
            completion_lookup = completion[:-i]

    except ValueError as e:
        log.warning(f"Failed to parse code: {e}")

    return completion


def strip_code_block_markdown(text: str) -> str:
    text = _RE_MARKDOWN_CODE_BLOCK_BEGIN.sub("", text, count=0)
    text = text.rstrip("`")

    return text


def prepend_new_line(code_context: str, completion: str) -> str:
    if (
        len(completion)
        and not code_context.endswith("\n")
        and not completion.startswith("\n")
    ):
        completion = "\n" + completion

    return completion


def _split_code_lines(s: str) -> list[str]:
    lines_split = s.splitlines(keepends=True)
    lines_processed = []

    for i, line in enumerate(lines_split):
        line = line.rstrip("\n")
        if i > 0:
            line = "\n" + line

        lines_processed.append(line)

    if len(lines_split) and lines_split[-1].endswith("\n"):
        lines_processed.append("\n")

    return lines_processed


# If the completion contains only comments, we should not return anything
def remove_comment_only_completion(
    completion: str,
    lang_id: Optional[LanguageId] = None,
) -> str:
    if not completion:
        return completion
    try:
        parser = CodeParser.from_language_id(
            completion,
            lang_id,
        )
        if parser.comments_only():
            log.info("removing comments-only completion")
            return ""
    except ValueError as e:
        log.warning(f"Failed to parse code: {e}")

    return completion
