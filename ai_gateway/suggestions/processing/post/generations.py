from typing import Any

from ai_gateway.suggestions.processing.post.base import PostProcessorBase
from ai_gateway.suggestions.processing.post.ops import (
    prepend_new_line,
    strip_code_block_markdown,
)

__all__ = [
    "PostProcessor",
]


class PostProcessor(PostProcessorBase):
    def __init__(self, code_context: str):
        self.code_context = code_context

    def process(self, completion: str, **kwargs: Any) -> str:
        completion = strip_code_block_markdown(completion)
        completion = prepend_new_line(self.code_context, completion)

        return completion
