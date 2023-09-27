from typing import Any

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

    def __init__(self, total_max_len: int, tkn_strategy: TokenStrategyBase):
        super().__init__(total_max_len, tkn_strategy)

        self.content: list[str] = []

        # This prompt builder requires a `prefix` placeholder to be present in the template
        self.tpl_args[PromptBuilderPrefixBased.KEY_PREFIX] = ""

    def add_content(self, *text: str, **_kwargs: Any):
        self.content.extend(text)

    def build(self) -> Prompt:
        prefix = self._build_prefix()
        prefix_with_tpl = self._apply_template(prefix.text)

        return Prompt(
            prefix=prefix_with_tpl,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(
                        length=len(prefix.text),
                        length_tokens=prefix.length_tokens,
                    ),
                }
            ),
        )

    def _build_prefix(self) -> CodeContent:
        prefix_snippets = []
        max_length = self.total_max_len - self.always_len

        for code in self.content:
            if max_length <= 0:
                break

            truncated = self.tkn_strategy.truncate_content(
                code, max_length, truncation_side="left"
            )

            max_length -= truncated.length_tokens
            prefix_snippets.append(truncated.text)

        return CodeContent(
            text="\n".join(prefix_snippets),
            # Take only the length of the prefix without template
            length_tokens=self.total_max_len - max_length - self.always_len,
        )

    def _apply_template(self, prefix: str) -> str:
        if self.tpl:
            self.tpl_args[PromptBuilderPrefixBased.KEY_PREFIX] = prefix
            return self.tpl.apply(**self.tpl_args)

        return prefix
