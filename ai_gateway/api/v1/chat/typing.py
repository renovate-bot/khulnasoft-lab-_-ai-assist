from typing import Annotated, List, Literal, Optional, Union

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from ai_gateway.models import (
    KindAnthropicModel,
    KindLiteLlmModel,
    KindModelProvider,
    Message,
)

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
    max_tokens_to_sample: Annotated[int, Field(ge=1, le=8192)] = 8192


class PromptPayload(BaseModel):
    content: Union[
        Annotated[str, StringConstraints(max_length=400000)],
        Annotated[list[Message], Field(min_length=1, max_length=100)],
    ]
    provider: Optional[
        Literal[KindModelProvider.ANTHROPIC, KindModelProvider.LITELLM]
    ] = None
    model: Optional[KindAnthropicModel | KindLiteLlmModel] = (
        KindAnthropicModel.CLAUDE_2_0
    )
    params: Optional[AnthropicParams] = None
    model_endpoint: Optional[str] = None
    model_api_key: Optional[str] = None
    model_identifier: Optional[str] = None


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
