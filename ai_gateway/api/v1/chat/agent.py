from time import time
from typing import Annotated, AsyncIterator, List, Literal, Optional

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.deps import ChatContainer
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicModel,
    KindAnthropicModel,
    KindModelProvider,
    TextGenModelChunk,
)

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("chat")

router = APIRouter(
    prefix="",
    tags=["chat"],
)


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


@router.post("/agent", response_model=ChatResponse)
@requires("duo_chat")
@feature_category("duo_chat")
@inject
async def chat(
    request: Request,
    chat_request: ChatRequest,
    anthropic_model: AnthropicModel = Depends(Provide[ChatContainer.anthropic_model]),
):
    prompt_component = chat_request.prompt_components[0]
    payload = prompt_component.payload

    anthropic_opts = {"model_name": payload.model.value}

    if payload.params:
        anthropic_opts.update(payload.params.dict())

    model = anthropic_model.provider(**anthropic_opts)

    try:
        if completion := await model.generate(
            prefix=payload.content,
            _suffix="",
            stream=chat_request.stream,
        ):
            if isinstance(completion, AsyncIterator):
                return await _handle_stream(completion)
            return ChatResponse(
                response=completion.text,
                metadata=ChatResponseMetadata(
                    provider=KindModelProvider.ANTHROPIC.value,
                    model=payload.model.value,
                    timestamp=int(time()),
                ),
            )
    except (AnthropicAPIConnectionError, AnthropicAPIStatusError) as ex:
        log.error(f"failed to execute Anthropic request: {ex}")
    return ChatResponse(
        response="",
        metadata=ChatResponseMetadata(
            provider=KindModelProvider.ANTHROPIC.value,
            model=payload.model.value,
            timestamp=int(time()),
        ),
    )


async def _handle_stream(
    response: AsyncIterator[TextGenModelChunk],
) -> StreamChatResponse:
    async def _stream_generator():
        async for result in response:
            yield result.text

    return StreamChatResponse(_stream_generator(), media_type="text/event-stream")
