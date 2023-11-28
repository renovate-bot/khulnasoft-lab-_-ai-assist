from time import time
from typing import Optional

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from pydantic.types import conlist, constr
from pydantic.typing import Literal
from starlette.authentication import requires

from ai_gateway.deps import ChatContainer
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicModel,
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
    source: constr(max_length=100)
    version: constr(max_length=100)


class PromptPayload(BaseModel):
    content: constr(max_length=400000)
    provider: Optional[
        Literal[AnthropicModel.MODEL_ENGINE]
    ]  # We only support and expect Anthropic for now
    model: Optional[
        Literal[AnthropicModel.CLAUDE, AnthropicModel.CLAUDE_INSTANT]
    ] = AnthropicModel.CLAUDE


class PromptComponent(BaseModel):
    type: constr(max_length=100)
    metadata: PromptMetadata
    payload: PromptPayload


# We expect only a single prompt component in the first iteration.
# Details: https://gitlab.com/gitlab-org/gitlab/-/merge_requests/135837#note_1642865693
class ChatRequest(BaseModel):
    prompt_components: conlist(PromptComponent, min_items=1, max_items=1)


class ChatResponseMetadata(BaseModel):
    provider: str
    model: str
    timestamp: int


class ChatResponse(BaseModel):
    response: str
    metadata: ChatResponseMetadata


@router.post("/agent", response_model=ChatResponse)
@requires("duo_chat")
@inject
async def chat(
    request: Request,
    chat_request: ChatRequest,
    anthropic_model: AnthropicModel = Depends(Provide[ChatContainer.anthropic_model]),
):
    prompt_component = chat_request.prompt_components[0]
    payload = prompt_component.payload
    model = anthropic_model.provider(model_name=payload.model)

    try:
        if completion := await model.generate(
            prefix=payload.content,
            _suffix="",
        ):
            return ChatResponse(
                response=completion.text,
                metadata=ChatResponseMetadata(
                    provider=AnthropicModel.MODEL_ENGINE,
                    model=payload.model,
                    timestamp=time(),
                ),
            )
    except (AnthropicAPIConnectionError, AnthropicAPIStatusError) as ex:
        log.error(f"failed to execute Anthropic request: {ex}")
    return ChatResponse(
        response="",
        metadata=ChatResponseMetadata(
            provider=AnthropicModel.MODEL_ENGINE, model=payload.model, timestamp=time()
        ),
    )
