from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from ai_gateway.chat.agents import Context, CurrentFile, Message
from ai_gateway.chat.agents.typing import AdditionalContext
from ai_gateway.models.base_chat import Message as LegacyMessage
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
    # Deprecated in favor of `messages`
    chat_history: Optional[
        Annotated[list[LegacyMessage] | list[str] | str, Field(deprecated=True)]
    ] = None
    agent_scratchpad: ReActAgentScratchpad = Field(discriminator="agent_type")
    # Deprecated in favor of `messages`
    context: Annotated[Optional[Context], Field(deprecated=True)] = None
    # Deprecated in favor of `messages`
    current_file: Annotated[Optional[CurrentFile], Field(deprecated=True)] = None
    # Deprecated in favor of `messages`
    additional_context: Annotated[
        Optional[list[AdditionalContext]], Field(deprecated=True)
    ] = None


class AgentRequest(BaseModel):
    # Deprecated in favor of `messages`
    prompt: Optional[Annotated[str, Field(deprecated=True)]] = None
    # TODO: Make this field required after `prompt` and `chat_history` are removed.
    # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/669
    messages: Optional[list[Message]] = None
    options: Optional[AgentRequestOptions] = None
    model_metadata: Optional[ModelMetadata] = None
    unavailable_resources: Optional[list[str]] = [
        "Merge Requests, Pipelines, Vulnerabilities"
    ]
