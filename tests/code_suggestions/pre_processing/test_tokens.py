from typing import Union

import pytest
from transformers import AutoTokenizer

from ai_gateway.code_suggestions.processing.pre import TokenizerTokenStrategy


class TestTokenizerTokenStrategy:
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-16B")

    @pytest.mark.parametrize(
        ("text", "max_length", "side", "text_expected"),
        [
            ("random_text", 2048, "left", "random_text"),
            ("random_text", 2048, "right", "random_text"),
            ("random_text", 0, "left", ""),
            ("random_text", 0, "right", ""),
            ("random_text", 1, "left", "text"),
            ("random_text", 1, "right", "random"),
        ],
    )
    def test_truncate_content(
        self, text: str, max_length: int, side: str, text_expected: str
    ):
        strategy = TokenizerTokenStrategy(self.tokenizer)

        actual = strategy.truncate_content(text, max_length, truncation_side=side)

        assert actual.text == text_expected
        if len(text_expected):
            assert actual.length_tokens > 0
        else:
            assert actual.length_tokens == 0

    @pytest.mark.parametrize(
        ("text", "expected_length"),
        [
            ("random_text", 3),
            (["random_text", "random"], [3, 1]),
            (["random_text", "random"], [3, 1]),
            ("", 0),
            (["", ""], [0, 0]),
        ],
    )
    def test_estimate_length(
        self, text: Union[str, list[str]], expected_length: Union[int, list[int]]
    ):
        strategy = TokenizerTokenStrategy(self.tokenizer)

        if isinstance(text, str):
            actual = strategy.estimate_length(text)
        else:
            actual = strategy.estimate_length(*text)

        assert actual == expected_length
