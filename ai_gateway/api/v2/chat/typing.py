from typing import Literal, Optional

from pydantic import BaseModel, Field

from ai_gateway.chat.typing import Context

__all__ = [
    "ReActAgentScratchpad",
    "ReActAgentAction",
    "AgentRequest",
    "AgentResponse",
]


class ReActAgentScratchpad(BaseModel):
    class AgentStep(BaseModel):
        thought: str
        tool: str
        tool_input: str
        observation: str

    agent_type: Literal["react"]
    steps: list[AgentStep]


class ReActAgentAction(BaseModel):
    class AgentToolAction(BaseModel):
        tool: str
        tool_input: str
        type: str = "tool"

    class AgentFinalAnswer(BaseModel):
        text: str
        type: str = "final_answer"

    thought: str
    step: AgentToolAction | AgentFinalAnswer


class AgentRequest(BaseModel):
    question: str
    chat_history: str | list[str]
    agent_scratchpad: ReActAgentScratchpad = Field(discriminator="agent_type")
    context: Optional[Context] = None


class AgentResponse(BaseModel):
    agent_action: ReActAgentAction
    log: Optional[str] = None
