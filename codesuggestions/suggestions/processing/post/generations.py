from typing import Any

from codesuggestions.suggestions.processing.post.base import PostProcessorBase
from codesuggestions.suggestions.processing.post.ops import strip_code_block_markdown

__all__ = [
    "PostProcessor",
]


class PostProcessor(PostProcessorBase):
    def process(self, completion: str, **kwargs: Any) -> str:
        completion = strip_code_block_markdown(completion)

        return completion
