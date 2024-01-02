from typing import Annotated, List, Literal, Optional

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from ai_gateway.models import KindAnthropicModel, KindModelProvider

__all__ = [
    "ChatRequest",
    "ChatResponseMetadata",
    "ChatResponse",
    "StreamChatResponse",
]


class PromptMetadata(BaseModel):
    source: Annotated[str, StringConstraints(max_length=100)]
    version: Annotated[str, StringConstraints(max_length=100)]


class AnthropicParams(BaseModel):
    stop_sequences: Annotated[
        List[Annotated[str, StringConstraints(max_length=225)]],
        Field(min_length=1, max_length=10),
    ] = [
        "\n\nHuman",
        "Observation:",
    ]
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    max_tokens_to_sample: Annotated[int, Field(ge=1, le=2_048)] = 2_048


class PromptPayload(BaseModel):
    content: Annotated[str, StringConstraints(max_length=400000)]
    provider: Optional[
        Literal[KindModelProvider.ANTHROPIC]
    ] = None  # We only support and expect Anthropic for now
    model: Optional[KindAnthropicModel] = KindAnthropicModel.CLAUDE_2_0
    params: Optional[AnthropicParams] = None


class PromptComponent(BaseModel):
    type: Annotated[str, StringConstraints(max_length=100)]
    metadata: PromptMetadata
    payload: PromptPayload


# We expect only a single prompt component in the first iteration.
# Details: https://gitlab.com/gitlab-org/gitlab/-/merge_requests/135837#note_1642865693
class ChatRequest(BaseModel):
    prompt_components: Annotated[
        List[PromptComponent], Field(min_length=1, max_length=1)
    ]
    stream: Optional[bool] = False


class ChatResponseMetadata(BaseModel):
    provider: str
    model: str
    timestamp: int


class ChatResponse(BaseModel):
    response: str
    metadata: ChatResponseMetadata


class StreamChatResponse(StreamingResponse):
    pass
