from pathlib import Path
from typing import Any

from ai_gateway.code_suggestions.prompts.base import PromptTemplateBase

__all__ = [
    "PromptTemplate",
    "PromptTemplateFewShot",
]


class PromptTemplate(PromptTemplateBase):
    def apply(self, **kwargs: Any) -> str:
        return self.tpl_raw.format(**kwargs)

    @classmethod
    def from_local_file(cls, filepath: Path):
        tpl_raw = cls._read_tpl_raw(filepath)
        return cls(tpl_raw)


class PromptTemplateFewShot(PromptTemplateBase):
    def __init__(
        self,
        tpl_raw: str,
        examples: list[dict],
        example_prompt: PromptTemplate,
        sep: str,
    ):
        pre_tpls = [example_prompt.apply(**example) for example in examples]

        tpl_raw = f"{sep}".join([*pre_tpls, tpl_raw])

        super().__init__(tpl_raw)

    def apply(self, **kwargs: Any) -> str:
        return self.tpl_raw.format(**kwargs)

    @classmethod
    def from_local_file(
        cls,
        filepath: Path,
        examples: list[dict],
        example_prompt: PromptTemplate,
        sep: str = "\n\n",
    ):
        tpl_raw = cls._read_tpl_raw(filepath)
        return cls(
            tpl_raw,
            examples,
            example_prompt,
            sep,
        )
