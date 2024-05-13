from time import time
from typing import AsyncIterator, Union

import structlog
from dependency_injector.providers import FactoryAggregate
from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.chat.typing import (
    ChatRequest,
    ChatResponse,
    ChatResponseMetadata,
    PromptPayload,
    StreamChatResponse,
)
from ai_gateway.async_dependency_resolver import (
    get_chat_anthropic_claude_factory_provider,
)
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicAPITimeoutError,
    KindModelProvider,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.tracking import log_exception

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("chat")

router = APIRouter()


@router.post("/agent", response_model=ChatResponse, status_code=status.HTTP_200_OK)
@requires(GitLabUnitPrimitive.DUO_CHAT)
@feature_category(GitLabFeatureCategory.DUO_CHAT)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    anthropic_claude_factory: FactoryAggregate = Depends(
        get_chat_anthropic_claude_factory_provider
    ),
):
    prompt_component = chat_request.prompt_components[0]
    payload = prompt_component.payload

    try:
        completion = await _generate_completion(
            anthropic_claude_factory, payload, stream=chat_request.stream
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


async def _generate_completion(
    anthropic_claude_factory: FactoryAggregate,
    prompt: PromptPayload,
    stream: bool = False,
) -> Union[TextGenModelOutput, AsyncIterator[TextGenModelChunk]]:
    opts = prompt.params.dict() if prompt.params else {}

    if isinstance(prompt.content, str):
        factory_type = (
            "llm"  # retrieve `AnthropicModel` from the FactoryAggregate object
        )
        opts.update({"prefix": prompt.content, "stream": stream})
    else:  # otherwise, `list[Message]`
        factory_type = (
            "chat"  # retrieve `AnthropicChatModel` from the FactoryAggregate object
        )
        opts.update({"messages": prompt.content, "stream": stream})

        # Hack: Anthropic renamed the `max_tokens_to_sample` arg to `max_tokens` for the new Message API
        if max_tokens := opts.pop("max_tokens_to_sample", None):
            opts["max_tokens"] = max_tokens

    completion = await anthropic_claude_factory(
        factory_type, name=prompt.model
    ).generate(**opts)

    return completion


async def _handle_stream(
    response: AsyncIterator[TextGenModelChunk],
) -> StreamChatResponse:
    async def _stream_generator():
        async for result in response:
            yield result.text

    return StreamChatResponse(_stream_generator(), media_type="text/event-stream")
