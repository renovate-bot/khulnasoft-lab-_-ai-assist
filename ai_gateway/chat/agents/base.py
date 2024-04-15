from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

__all__ = [
    "TypeAgentAction",
    "TypeAgentInputs",
    "AgentMessage",
    "AgentToolAction",
    "AgentFinalAnswer",
    "AgentStep",
    "BaseParser",
    "BaseMultiStepAgent",
]


class AgentMessage(BaseModel):
    log: Optional[str] = None


class AgentToolAction(AgentMessage):
    tool: str
    tool_input: str


class AgentFinalAnswer(AgentMessage):
    text: str


TypeAgentAction = TypeVar("TypeAgentAction", bound=AgentToolAction | AgentFinalAnswer)
TypeAgentInputs = TypeVar("TypeAgentInputs")
_TypeParserOutput = TypeVar("_TypeParserOutput")


class AgentStep(BaseModel, Generic[TypeAgentAction]):
    action: TypeAgentAction
    observation: str


class BaseParser(ABC, Generic[_TypeParserOutput]):
    @abstractmethod
    def parse(self, text: str) -> _TypeParserOutput:
        pass


class BaseMultiStepAgent(ABC, BaseModel, Generic[TypeAgentInputs, TypeAgentAction]):
    agent_scratchpad: list[AgentStep[TypeAgentAction]] = Field(
        default_factory=lambda: []
    )

    @abstractmethod
    async def invoke(
        self, *, inputs: TypeAgentInputs, **kwargs: Any
    ) -> TypeAgentAction:
        pass
