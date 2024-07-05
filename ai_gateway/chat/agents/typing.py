from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel

__all__ = [
    "AgentMessage",
    "AgentToolAction",
    "AgentFinalAnswer",
    "AgentStep",
    "TypeAgentInputs",
    "TypeAgentAction",
    "Context",
    "CurrentFileContext",
]


class AgentMessage(BaseModel):
    log: Optional[str] = None


class AgentToolAction(AgentMessage):
    tool: str
    tool_input: str


class AgentFinalAnswer(AgentMessage):
    text: str


TypeAgentInputs = TypeVar("TypeAgentInputs")
TypeAgentAction = TypeVar("TypeAgentAction", bound=AgentToolAction | AgentFinalAnswer)


class AgentStep(BaseModel, Generic[TypeAgentAction]):
    action: TypeAgentAction
    observation: str


class Context(BaseModel, frozen=True):  # type: ignore[call-arg]
    type: Literal["issue", "epic"]
    content: str


class CurrentFileContext(BaseModel):
    file_name: str
    selected_text: str
