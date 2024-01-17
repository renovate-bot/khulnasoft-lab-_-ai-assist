from functools import partial
from typing import Any, Callable, NewType, Optional

from ai_gateway.code_suggestions.processing.ops import strip_whitespaces
from ai_gateway.code_suggestions.processing.post.base import PostProcessorBase
from ai_gateway.code_suggestions.processing.post.ops import (
    clean_model_reflection,
    fix_end_block_errors,
    remove_comment_only_completion,
    trim_by_min_allowed_context,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId

__all__ = [
    "PostProcessor",
]


AliasOpsRecord = NewType("AliasOpsRecord", tuple[str, Callable[[str], str]])


class PostProcessor(PostProcessorBase):
    def __init__(
        self,
        code_context: str,
        lang_id: Optional[LanguageId] = None,
        suffix: Optional[str] = None,
        exclude: Optional[list] = None,
    ):
        self.code_context = code_context
        self.lang_id = lang_id
        self.suffix = suffix if suffix else ""
        self.exclude = set(exclude) if exclude else []

    @property
    def ops(self) -> list[AliasOpsRecord]:
        return [
            (
                remove_comment_only_completion.__name__,
                partial(remove_comment_only_completion, lang_id=self.lang_id),
            ),
            (
                trim_by_min_allowed_context.__name__,
                partial(
                    trim_by_min_allowed_context, self.code_context, lang_id=self.lang_id
                ),
            ),
            (
                fix_end_block_errors.__name__,
                partial(
                    fix_end_block_errors,
                    self.code_context,
                    suffix=self.suffix,
                    lang_id=self.lang_id,
                ),
            ),
            (
                clean_model_reflection.__name__,
                partial(clean_model_reflection, self.code_context),
            ),
            (strip_whitespaces.__name__, strip_whitespaces),
        ]

    async def process(self, completion: str, **kwargs: Any) -> str:
        for key, func in self.ops:
            if key in self.exclude:
                continue

            completion = await func(completion)
            if completion == "":
                return ""

        return completion
