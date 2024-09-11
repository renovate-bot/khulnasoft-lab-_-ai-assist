from typing import Literal, Optional

from pydantic import BaseModel, Field

from ai_gateway.chat.agents import Context, CurrentFile
from ai_gateway.chat.agents.typing import AdditionalContext
from ai_gateway.prompts.typing import ModelMetadata

__all__ = [
    "ReActAgentScratchpad",
    "AgentRequestOptions",
    "AgentRequest",
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
    additional_context: Optional[list[AdditionalContext]] = None


class AgentRequest(BaseModel):
    prompt: str
    options: AgentRequestOptions
    model_metadata: Optional[ModelMetadata] = None
    unavailable_resources: Optional[list[str]] = [
        "Merge Requests, Pipelines, Vulnerabilities"
    ]
