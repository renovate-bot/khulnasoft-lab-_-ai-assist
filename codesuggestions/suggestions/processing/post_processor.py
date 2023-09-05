__all__ = [
    "PostProcessor",
    "strip_whitespaces",
    "clean_model_reflection",
]

from codesuggestions.suggestions.processing.ops import (
    find_common_lines,
    find_newline_position,
)

_COMMENT_IDENTIFIERS = ["/*", "//", "#"]


class PostProcessor:
    def process(self, code_context: str, text: str) -> str:
        """Execute a number of post-processing actions on a text.

        Args:
           text: A text to process

        Returns:
           A resulted output
        """

        text = clean_model_reflection(code_context, text)
        text = strip_whitespaces(text)

        return text


def strip_whitespaces(text: str) -> str:
    return "" if text.isspace() else text


def clean_model_reflection(context: str, completion: str) -> str:
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

        if len(group) == 1 and not target_lines[0].lstrip().startswith(
            tuple(_COMMENT_IDENTIFIERS)
        ):
            # This line doesn't look like a comment, no need to dedup
            lines_completion.append(target_lines[0])

        prev_line = end_line + 1

    # Add remaining lines to the completion list
    lines_completion.extend(lines_after[prev_line:])

    # Get the completion of the current line + processed lines
    completion = text[len(context) : br_pos]
    completion = "".join([completion, *lines_completion])

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
