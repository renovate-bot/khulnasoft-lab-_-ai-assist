from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, NamedTuple, Union

from ai_gateway.code_suggestions.processing import (
    MetadataCodeContent,
    MetadataPromptBuilder,
    Prompt,
)
from ai_gateway.prompts import PromptTemplateBase

__all__ = [
    "CodeKind",
    "CodeContent",
    "TokenStrategyBase",
    "PromptBuilderBase",
]


class CodeKind(Enum):
    PREFIX = 1
    SUFFIX = 2
    SNIPPET = 3


class CodeContent(NamedTuple):
    text: str
    length_tokens: int


class TokenStrategyBase(ABC):
    @abstractmethod
    def truncate_content(
        self, text: str, max_length: int, truncation_side: str = "left"
    ) -> CodeContent:
        pass

    @abstractmethod
    def estimate_length(self, *text: str) -> Union[int, list[int]]:
        pass


class PromptBuilderBase(ABC):
    def __init__(self, total_max_len: int, tkn_strategy: TokenStrategyBase):
        self.total_max_len = max(total_max_len, 0)
        self.tkn_strategy = tkn_strategy

        self.always_len = 0

        self.tpl: PromptTemplateBase = None
        self.tpl_args = dict()

    def add_template(self, tpl: PromptTemplateBase, **kwargs: Any) -> int:
        self.tpl = tpl
        self.tpl_args.update(kwargs)

        # Apply all known arguments to get the number of reserved tokens
        tpl_raw = self.tpl.apply(**self.tpl_args)
        tpl_len = self.tkn_strategy.estimate_length(tpl_raw)
        self.always_len += tpl_len

        return tpl_len

    def wrap(self, prompt: str, ignore_exception: bool = False) -> Prompt:
        length_tokens = self.tkn_strategy.estimate_length(prompt)
        if length_tokens > self.total_max_len and not ignore_exception:
            raise ValueError("the prompt size exceeds overall maximum length")

        return Prompt(
            prefix=prompt,
            metadata=MetadataPromptBuilder(
                components={
                    "prompt": MetadataCodeContent(
                        length=len(prompt),
                        length_tokens=length_tokens,
                    ),
                }
            ),
        )

    @abstractmethod
    def add_content(self, *text: str, **_kwargs: Any):
        pass

    @abstractmethod
    def build(self) -> Prompt:
        pass
