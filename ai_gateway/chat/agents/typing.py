from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel

__all__ = [
    "AgentToolAction",
    "AgentFinalAnswer",
    "AgentStep",
    "TypeAgentInputs",
    "TypeAgentAction",
    "Context",
    "CurrentFile",
    "AdditionalContext",
]


class AgentToolAction(BaseModel):
    thought: str
    tool: str
    tool_input: str


class AgentFinalAnswer(BaseModel):
    text: str


TypeAgentInputs = TypeVar("TypeAgentInputs")
TypeAgentAction = TypeVar("TypeAgentAction", bound=AgentToolAction | AgentFinalAnswer)


class AgentStep(BaseModel, Generic[TypeAgentAction]):
    action: TypeAgentAction
    observation: str


class Context(BaseModel, frozen=True):  # type: ignore[call-arg]
    type: Literal["issue", "epic", "merge_request"]
    content: str


class CurrentFile(BaseModel):
    file_path: str
    data: str
    selected_code: bool


# Note: additionaL_context is an alias for injected_context
class AdditionalContext(BaseModel):
    id: Optional[str] = None
    category: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict] = None
