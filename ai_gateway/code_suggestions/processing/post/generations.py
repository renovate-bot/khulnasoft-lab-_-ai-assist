from typing import Any

from ai_gateway.code_suggestions.processing.ops import strip_whitespaces
from ai_gateway.code_suggestions.processing.post.base import PostProcessorBase
from ai_gateway.code_suggestions.processing.post.ops import (
    clean_model_reflection,
    prepend_new_line,
    strip_code_block_markdown,
)

__all__ = ["PostProcessor", "PostProcessorAnthropic"]


class PostProcessor(PostProcessorBase):
    def __init__(self, code_context: str):
        self.code_context = code_context

    def process(self, completion: str, **kwargs: Any) -> str:
        completion = strip_code_block_markdown(completion)
        completion = prepend_new_line(self.code_context, completion)

        # Note: `clean_model_reflection` joins code context and completion
        # we need to call the function right after prepending a new line
        completion = clean_model_reflection(self.code_context, completion)
        completion = strip_whitespaces(completion)

        return completion


class PostProcessorAnthropic(PostProcessor):
    def process(self, completion: str, **kwargs: Any) -> str:
        completion = strip_whitespaces(completion)

        return completion
