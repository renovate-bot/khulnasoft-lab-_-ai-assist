from typing import Literal, Optional

from pydantic import BaseModel, Field

from ai_gateway.chat.agents import Message
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
    agent_scratchpad: ReActAgentScratchpad = Field(discriminator="agent_type")


class AgentRequest(BaseModel):
    messages: list[Message]
    options: Optional[AgentRequestOptions] = None
    model_metadata: Optional[ModelMetadata] = None
    unavailable_resources: Optional[list[str]] = [
        "Merge Requests, Pipelines, Vulnerabilities"
    ]
