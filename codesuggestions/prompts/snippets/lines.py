from typing import Iterable

from codesuggestions.prompts.snippets import CodeSnippet
from codesuggestions.prompts.snippets.base import BaseCodeSnippetsIterator

__all__ = ["LineBasedCodeSnippets"]


class LineBasedCodeSnippets(BaseCodeSnippetsIterator):
    def __init__(self, content: str, num_lines: int, drop_last: bool = False):
        super().__init__(content)

        self.num_lines = num_lines
        self.drop_last = drop_last

    def _next_snippet(self) -> Iterable[CodeSnippet]:
        content_lines = self.content.splitlines(keepends=False)
        for i in range(0, len(content_lines), self.num_lines):
            snippet_lines = content_lines[i : i + self.num_lines]
            snippet_lines_len = len(snippet_lines)

            if snippet_lines_len < self.num_lines and self.drop_last:
                continue

            yield CodeSnippet(
                text="\n".join(snippet_lines),
                start_pos=(i, 0),
                end_pos=(i + snippet_lines_len - 1, len(snippet_lines[-1])),
            )
