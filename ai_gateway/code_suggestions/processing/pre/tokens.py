from typing import Union

from transformers import PreTrainedTokenizer

from ai_gateway.code_suggestions.processing.pre.base import (
    CodeContent,
    TokenStrategyBase,
)

__all__ = [
    "TokenizerTokenStrategy",
]


class TokenizerTokenStrategy(TokenStrategyBase):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def truncate_content(
        self, text: str, max_length: int, truncation_side: str = "left"
    ) -> CodeContent:
        prev_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = truncation_side

        tokens = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        decoded = self.tokenizer.decode(tokens["input_ids"])
        self.tokenizer.truncation_side = prev_truncation_side

        return CodeContent(
            text=decoded,
            length_tokens=len(tokens["input_ids"]),
        )

    def estimate_length(self, *text: str) -> Union[int, list[int]]:
        lengths = self.tokenizer(
            list(text),
            return_length=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )["length"]

        if len(lengths) == 1:
            return lengths.pop()

        return lengths
