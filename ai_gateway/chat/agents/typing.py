import json
from typing import Literal, Optional, TypeVar

from pydantic import BaseModel

__all__ = [
    "AgentToolAction",
    "AgentFinalAnswer",
    "AgentUnknownAction",
    "AgentBaseEvent",
    "TypeAgentEvent",
    "AgentStep",
    "TypeAgentInputs",
    "Context",
    "CurrentFile",
    "AdditionalContext",
]


class AgentBaseEvent(BaseModel):
    def dump_as_response(self) -> str:
        model_dump = self.model_dump()
        type = model_dump.pop("type")
        return json.dumps({"type": type, "data": model_dump})


class AgentToolAction(AgentBaseEvent):
    type: str = "action"
    tool: str
    tool_input: str
    thought: str


class AgentFinalAnswer(AgentBaseEvent):
    type: str = "final_answer_delta"
    text: str


class AgentUnknownAction(AgentBaseEvent):
    type: str = "unknown"
    text: str


TypeAgentEvent = TypeVar(
    "TypeAgentEvent", AgentToolAction, AgentFinalAnswer, AgentUnknownAction
)

TypeAgentInputs = TypeVar("TypeAgentInputs")


class AgentStep(BaseModel):
    action: AgentToolAction
    observation: str


class Context(BaseModel, frozen=True):  # type: ignore[call-arg]
    type: Literal["issue", "epic", "merge_request", "commit"]
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
