from enum import StrEnum
from functools import partial
from inspect import iscoroutinefunction
from typing import Any, Callable, NewType, Optional

from ai_gateway.code_suggestions.processing.ops import strip_whitespaces
from ai_gateway.code_suggestions.processing.post.base import PostProcessorBase
from ai_gateway.code_suggestions.processing.post.ops import (
    clean_model_reflection,
    fix_end_block_errors,
    fix_end_block_errors_with_comparison,
    remove_comment_only_completion,
    strip_asterisks,
    trim_by_min_allowed_context,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId

__all__ = [
    "PostProcessorOperation",
    "PostProcessor",
]


AliasOpsRecord = NewType("AliasOpsRecord", tuple[str, Callable[[str], str]])


class PostProcessorOperation(StrEnum):
    REMOVE_COMMENTS = "remove_comment_only_completion"
    TRIM_BY_MINIMUM_CONTEXT = "trim_by_min_allowed_context"
    FIX_END_BLOCK_ERRORS = "fix_end_block_errors"
    FIX_END_BLOCK_ERRORS_WITH_COMPARISON = "fix_end_block_errors_with_comparison"
    CLEAN_MODEL_REFLECTION = "clean_model_reflection"
    STRIP_WHITESPACES = "strip_whitespaces"
    STRIP_ASTERISKS = "strip_asterisks"


# This is the ordered list of prost-processing functions
# Please do not change the order unless you have determined that it is acceptable
ORDERED_POST_PROCESSORS = [
    PostProcessorOperation.REMOVE_COMMENTS,
    PostProcessorOperation.TRIM_BY_MINIMUM_CONTEXT,
    PostProcessorOperation.FIX_END_BLOCK_ERRORS,
    PostProcessorOperation.CLEAN_MODEL_REFLECTION,
    PostProcessorOperation.STRIP_WHITESPACES,
]


class PostProcessor(PostProcessorBase):
    def __init__(
        self,
        code_context: str,
        lang_id: Optional[LanguageId] = None,
        suffix: Optional[str] = None,
        overrides: Optional[
            dict[PostProcessorOperation, PostProcessorOperation]
        ] = None,
        exclude: Optional[list] = None,
        extras: Optional[list] = None,
    ):
        self.code_context = code_context
        self.lang_id = lang_id
        self.suffix = suffix if suffix else ""
        self.overrides = overrides if overrides else {}
        self.exclude = set(exclude) if exclude else []
        self.extras = extras if extras else []

    @property
    def ops(self) -> list[AliasOpsRecord]:
        return {
            PostProcessorOperation.REMOVE_COMMENTS: partial(
                remove_comment_only_completion, lang_id=self.lang_id
            ),
            PostProcessorOperation.TRIM_BY_MINIMUM_CONTEXT: partial(
                trim_by_min_allowed_context, self.code_context, lang_id=self.lang_id
            ),
            PostProcessorOperation.FIX_END_BLOCK_ERRORS: partial(
                fix_end_block_errors,
                self.code_context,
                suffix=self.suffix,
                lang_id=self.lang_id,
            ),
            PostProcessorOperation.FIX_END_BLOCK_ERRORS_WITH_COMPARISON: partial(
                fix_end_block_errors_with_comparison,
                self.code_context,
                suffix=self.suffix,
                lang_id=self.lang_id,
            ),
            PostProcessorOperation.CLEAN_MODEL_REFLECTION: partial(
                clean_model_reflection, self.code_context
            ),
            PostProcessorOperation.STRIP_WHITESPACES: strip_whitespaces,
            PostProcessorOperation.STRIP_ASTERISKS: strip_asterisks,
        }

    async def process(self, completion: str, **kwargs: Any) -> str:
        for processor in self._ordered_post_processors():
            if str(processor) in self.exclude:
                continue

            completion = await self._apply_post_processor(processor, completion)

            if completion == "":
                return ""

        return completion

    def _ordered_post_processors(self):
        return ORDERED_POST_PROCESSORS + self.extras

    async def _apply_post_processor(self, processor_key, completion):
        # Override post-processor if present in `overrides`, else use the given processor
        actual_processor_key = self.overrides.get(processor_key, processor_key)
        func = self.ops[actual_processor_key]

        if self._is_async(func):
            return await func(completion)

        return func(completion)

    def _is_async(self, func):
        return iscoroutinefunction(func)
