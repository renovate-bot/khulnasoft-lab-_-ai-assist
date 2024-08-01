import math
from typing import Any, Optional

from ai_gateway.code_suggestions.processing.pre.base import PromptBuilderBase
from ai_gateway.code_suggestions.processing.typing import (
    CodeContent,
    MetadataCodeContent,
    MetadataExtraInfo,
    MetadataPromptBuilder,
    Prompt,
    TokenStrategyBase,
)

__all__ = [
    "PromptBuilderPrefixBased",
]


class PromptBuilderPrefixBased(PromptBuilderBase):
    KEY_PREFIX = "prefix"
    KEY_SUFFIX = "suffix"
    KEY_CODE_CONTEXT = "code_context"

    # Percentage of tokens reserved for suffix (0 <= value <= 1).
    KEY_SUFFIX_RESERVED_PERCENT = "suffix_reserved_percent"
    DEFAULT_SUFFIX_RESERVED_PERCENT = 0

    def __init__(self, total_max_len: int, tkn_strategy: TokenStrategyBase):
        super().__init__(total_max_len, tkn_strategy)

        self.snippets: list[str] = []
        self.suffix: Optional[str] = None
        self.code_context: Optional[list] = None
        self.opts: dict = {
            self.KEY_SUFFIX_RESERVED_PERCENT: self.DEFAULT_SUFFIX_RESERVED_PERCENT
        }

        # This prompt builder requires a `prefix` placeholder to be present in the template
        self.tpl_args[self.KEY_PREFIX] = ""

    def add_content(
        self,
        *text: str,
        **kwargs: Any,
    ):
        self.snippets.extend(text)
        self.suffix = kwargs.pop(self.KEY_SUFFIX, None) or self.suffix
        self.code_context = kwargs.pop(self.KEY_CODE_CONTEXT, None) or self.code_context
        opts: dict = {}
        if dist := kwargs.pop(self.KEY_SUFFIX_RESERVED_PERCENT, None):
            opts[self.KEY_SUFFIX_RESERVED_PERCENT] = max(0, min(dist, 1))

        self.opts.update(opts)

    def build(self) -> Prompt:
        suffix_reserved_percent = self.opts[self.KEY_SUFFIX_RESERVED_PERCENT]
        max_length = self.total_max_len - self.always_len
        max_length_prefix = math.ceil((1 - suffix_reserved_percent) * max_length)
        max_length_suffix = max_length - max_length_prefix

        prefix = self._build_prefix(max_length_prefix)
        suffix = self._build_suffix(max_length_suffix)
        prefix_with_tpl = self._apply_template(prefix.text)

        max_length_code_context = max_length - prefix.length_tokens

        if suffix:
            max_length_code_context -= suffix.length_tokens
        code_context_info = self._build_code_context(max_length_code_context)

        if code_context_info and max_length_code_context > 0:
            _, truncated = code_context_info
            prefix_with_tpl = "\n".join([truncated.text, prefix_with_tpl])

        components = {
            name: MetadataCodeContent(
                length=len(component.text),
                length_tokens=component.length_tokens,
            )
            for name, component in zip(["prefix", "suffix"], [prefix, suffix])
            if component is not None
        }

        code_context_metadata = self._build_code_context_metadata(code_context_info)

        return Prompt(
            prefix=prefix_with_tpl,
            suffix=suffix.text if suffix else None,
            metadata=MetadataPromptBuilder(
                components=components,
                code_context=code_context_metadata,
            ),
        )

    def _build_prefix(self, max_length: int) -> CodeContent:
        prefix_snippets = []
        length = max_length

        for code in self.snippets:
            if length <= 0:
                break

            truncated = self.tkn_strategy.truncate_content(
                code, length, truncation_side="left"
            )

            length -= truncated.length_tokens
            prefix_snippets.append(truncated.text)

        return CodeContent(
            text="\n".join(prefix_snippets),
            # Take only the length of the prefix without template
            length_tokens=max_length - length,
        )

    def _build_suffix(self, max_length: int) -> Optional[CodeContent]:
        if not self.suffix:
            return None

        truncated = self.tkn_strategy.truncate_content(
            self.suffix, max_length, truncation_side="right"
        )

        return truncated

    def _build_code_context(
        self, max_length: int
    ) -> Optional[tuple[CodeContent, CodeContent]]:
        if not self.code_context:
            return None

        original = CodeContent(
            text="\n".join(self.code_context),
            length_tokens=self.tkn_strategy.estimate_length(self.code_context)[0],
        )

        truncated = self.tkn_strategy.truncate_content(
            original.text, max_length, truncation_side="right"
        )

        return original, truncated

    def _build_code_context_metadata(
        self, code_context_info: Optional[tuple[CodeContent, CodeContent]]
    ) -> Optional[MetadataExtraInfo]:
        if not code_context_info:
            return None

        original, truncated = code_context_info

        return MetadataExtraInfo(
            name="code_context",
            pre=MetadataCodeContent(
                length=len(original.text),
                length_tokens=original.length_tokens,
            ),
            post=MetadataCodeContent(
                length=len(truncated.text),
                length_tokens=truncated.length_tokens,
            ),
        )

    def _apply_template(self, prefix: str) -> str:
        if self.tpl:
            self.tpl_args[PromptBuilderPrefixBased.KEY_PREFIX] = prefix
            return self.tpl.apply(**self.tpl_args)

        return prefix
