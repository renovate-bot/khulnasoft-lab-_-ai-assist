from typing import Literal, Optional

from pydantic import BaseModel, Field

from ai_gateway.chat.agents import Context, CurrentFile, TypeReActAgentAction
from ai_gateway.prompts.typing import ModelMetadata

__all__ = [
    "ReActAgentScratchpad",
    "AgentRequestOptions",
    "AgentRequest",
    "AgentStreamResponseEvent",
]


class ReActAgentScratchpad(BaseModel):
    class AgentStep(BaseModel):
        thought: str
        tool: str
        tool_input: str
        observation: str

    agent_type: Literal["react"]
    steps: list[AgentStep]


class AgentRequestOptions(BaseModel):
    chat_history: str | list[str]
    agent_scratchpad: ReActAgentScratchpad = Field(discriminator="agent_type")
    context: Optional[Context] = None
    current_file: Optional[CurrentFile] = None


class AgentRequest(BaseModel):
    prompt: str
    options: AgentRequestOptions
    model_metadata: Optional[ModelMetadata] = None


class AgentStreamResponseEvent(BaseModel):
    type: Literal["action", "final_answer_delta"]
    data: TypeReActAgentAction
