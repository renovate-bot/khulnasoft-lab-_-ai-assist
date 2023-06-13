from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = [
    "PromptTemplateBase",
]


class PromptTemplateBase(ABC):
    def __init__(self, tpl_raw: str):
        self.tpl_raw = tpl_raw

    @property
    def raw(self):
        return self.tpl_raw

    @abstractmethod
    def apply(self, **kwargs: Any) -> str:
        pass

    @staticmethod
    def _read_tpl_raw(filepath: Path) -> str:
        with open(str(filepath), encoding="utf-8") as file:
            return file.read()
