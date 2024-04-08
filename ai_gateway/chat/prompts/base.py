from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

__all__ = ["ChatPrompt", "BasePromptRegistry"]


class ChatPrompt(BaseModel):
    user: str
    system: Optional[str] = None
    assistant: Optional[str] = None


class BasePromptRegistry(ABC):
    @abstractmethod
    def get_prompt(self, key: str, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def get_chat_prompt(self, key: str, **kwargs) -> ChatPrompt:
        pass
