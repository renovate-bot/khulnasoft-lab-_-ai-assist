from typing import Any, Optional

from ai_gateway.suggestions.processing.ops import strip_whitespaces
from ai_gateway.suggestions.processing.post.base import PostProcessorBase
from ai_gateway.suggestions.processing.post.ops import (
    clean_model_reflection,
    trim_by_min_allowed_context,
)
from ai_gateway.suggestions.processing.typing import LanguageId

__all__ = [
    "PostProcessor",
]


class PostProcessor(PostProcessorBase):
    def __init__(self, code_context: str, lang_id: Optional[LanguageId] = None):
        self.code_context = code_context
        self.lang_id = lang_id

    def process(self, completion: str, **kwargs: Any) -> str:
        completion = trim_by_min_allowed_context(
            self.code_context, completion, lang_id=self.lang_id
        )
        completion = clean_model_reflection(self.code_context, completion)
        completion = strip_whitespaces(completion)

        return completion
