from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

__all__ = [
    "AgentMessage",
    "AgentAction",
    "AgentFinalAnswer",
    "AgentStep",
    "BaseParser",
    "BaseMultiStepAgent",
]


class AgentMessage(BaseModel):
    log: Optional[str] = None


class AgentAction(AgentMessage):
    tool: str
    tool_input: str


class AgentFinalAnswer(AgentMessage):
    text: str


_U = TypeVar("_U", bound=AgentAction | AgentFinalAnswer)
_T = TypeVar("_T")
_P = TypeVar("_P")


class AgentStep(BaseModel, Generic[_U]):
    action: _U
    observation: str


class BaseParser(ABC, Generic[_P]):
    @abstractmethod
    def parse(self, text: str) -> _P:
        pass


class BaseMultiStepAgent(ABC, BaseModel, Generic[_T, _U]):
    agent_scratchpad: list[AgentStep[_U]] = Field(default_factory=lambda: [])

    @abstractmethod
    async def invoke(self, inputs: _T, **kwargs: Any) -> _U:
        pass
