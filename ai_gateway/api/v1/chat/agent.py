from time import time
from typing import AsyncIterator

import structlog
from dependency_injector.providers import Factory
from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.chat.typing import (
    ChatRequest,
    ChatResponse,
    ChatResponseMetadata,
    StreamChatResponse,
)
from ai_gateway.async_dependency_resolver import (
    get_chat_anthropic_claude_factory_provider,
)
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicAPITimeoutError,
    AnthropicModel,
    KindModelProvider,
    TextGenModelChunk,
)
from ai_gateway.tracking import log_exception

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("chat")

router = APIRouter()


@router.post("/agent", response_model=ChatResponse, status_code=status.HTTP_200_OK)
@requires("duo_chat")
@feature_category("duo_chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    anthropic_claude_factory: Factory[AnthropicModel] = Depends(
        get_chat_anthropic_claude_factory_provider
    ),
):
    prompt_component = chat_request.prompt_components[0]
    payload = prompt_component.payload

    anthropic_opts = {"name": payload.model}

    if payload.params:
        anthropic_opts.update(payload.params.dict())

    model = anthropic_claude_factory(**anthropic_opts)

    try:
        completion = await model.generate(
            prefix=payload.content,
            _suffix="",
            stream=chat_request.stream,
        )

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
    except AnthropicAPIStatusError as ex:
        log_exception(ex)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Anthropic API Status Error.",
        )
    except AnthropicAPITimeoutError as ex:
        log_exception(ex)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Anthropic API Timeout Error.",
        )
    except AnthropicAPIConnectionError as ex:
        log_exception(ex)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Anthropic API Connection Error.",
        )


async def _handle_stream(
    response: AsyncIterator[TextGenModelChunk],
) -> StreamChatResponse:
    async def _stream_generator():
        async for result in response:
            yield result.text

    return StreamChatResponse(_stream_generator(), media_type="text/event-stream")
