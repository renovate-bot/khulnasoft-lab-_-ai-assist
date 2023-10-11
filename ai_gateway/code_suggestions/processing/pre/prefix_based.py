import math
from typing import Any, Optional

from ai_gateway.code_suggestions.processing import (
    MetadataCodeContent,
    MetadataPromptBuilder,
    Prompt,
)
from ai_gateway.code_suggestions.processing.pre.base import (
    CodeContent,
    PromptBuilderBase,
    TokenStrategyBase,
)

__all__ = [
    "PromptBuilderPrefixBased",
]


class PromptBuilderPrefixBased(PromptBuilderBase):
    KEY_PREFIX = "prefix"
    KEY_SUFFIX = "suffix"

    # Percentage of tokens reserved for suffix (0 <= value <= 1).
    KEY_SUFFIX_RESERVED_PERCENT = "suffix_reserved_percent"
    DEFAULT_SUFFIX_RESERVED_PERCENT = 0

    def __init__(self, total_max_len: int, tkn_strategy: TokenStrategyBase):
        super().__init__(total_max_len, tkn_strategy)

        self.snippets: list[str] = []
        self.suffix: Optional[str] = None
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

        components = {
            name: MetadataCodeContent(
                length=len(component.text),
                length_tokens=component.length_tokens,
            )
            for name, component in zip(["prefix", "suffix"], [prefix, suffix])
            if component is not None
        }

        return Prompt(
            prefix=prefix_with_tpl,
            suffix=suffix.text if suffix else None,
            metadata=MetadataPromptBuilder(components=components),
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

    def _build_suffix(self, max_length: float) -> Optional[CodeContent]:
        if not self.suffix:
            return None

        truncated = self.tkn_strategy.truncate_content(
            self.suffix, max_length, truncation_side="right"
        )

        return truncated

    def _apply_template(self, prefix: str) -> str:
        if self.tpl:
            self.tpl_args[PromptBuilderPrefixBased.KEY_PREFIX] = prefix
            return self.tpl.apply(**self.tpl_args)

        return prefix
