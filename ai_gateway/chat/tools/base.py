from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

__all__ = [
    "BaseTool",
    "BaseRemoteTool",
    "BaseToolkit",
]


class BaseTool(ABC, BaseModel, frozen=True):
    name: str
    description: str
    resource: Optional[str] = None
    example: Optional[str] = None


class BaseRemoteTool(BaseTool):
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        # By default, we run tools on the Ruby app side
        raise NotImplementedError(
            "Please check the Rails app for an implementation of this tool."
        )


class BaseToolkit(ABC):
    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        raise NotImplementedError
