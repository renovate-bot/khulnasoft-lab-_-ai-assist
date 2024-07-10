from time import time
from typing import Annotated, AsyncIterator, Union

import structlog
from dependency_injector.providers import Factory, FactoryAggregate
from fastapi import APIRouter, Depends, HTTPException, Request, status

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.chat.auth import ChatInvokable, authorize_with_unit_primitive
from ai_gateway.api.v1.chat.typing import (
    ChatRequest,
    ChatResponse,
    ChatResponseMetadata,
    PromptPayload,
    StreamChatResponse,
)
from ai_gateway.async_dependency_resolver import (
    get_chat_anthropic_claude_factory_provider,
    get_chat_litellm_factory_provider,
)
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicAPITimeoutError,
    KindModelProvider,
)
from ai_gateway.models.base_text import TextGenModelChunk, TextGenModelOutput
from ai_gateway.tracking import log_exception

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("chat")

router = APIRouter()

CHAT_INVOKABLES = [
    ChatInvokable(name="explain_code", unit_primitive=GitLabUnitPrimitive.DUO_CHAT),
    ChatInvokable(name="write_tests", unit_primitive=GitLabUnitPrimitive.DUO_CHAT),
    ChatInvokable(name="refactor_code", unit_primitive=GitLabUnitPrimitive.DUO_CHAT),
    ChatInvokable(
        name="explain_vulnerability",
        unit_primitive=GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
    ),
    ChatInvokable(
        name="summarize_comments",
        unit_primitive=GitLabUnitPrimitive.SUMMARIZE_COMMENTS,
    ),
    # Deprecated. Added for backward compatibility.
    # Please, refer to `v2/chat/agent` for additional details.
    ChatInvokable(name="agent", unit_primitive=GitLabUnitPrimitive.DUO_CHAT),
]


@router.post(
    "/{chat_invokable}", response_model=ChatResponse, status_code=status.HTTP_200_OK
)
@feature_category(GitLabFeatureCategory.DUO_CHAT)
@authorize_with_unit_primitive("chat_invokable", chat_invokables=CHAT_INVOKABLES)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    chat_invokable: str,
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
    anthropic_claude_factory: FactoryAggregate = Depends(
        get_chat_anthropic_claude_factory_provider
    ),
    litellm_factory: Factory = Depends(get_chat_litellm_factory_provider),
):
    prompt_component = chat_request.prompt_components[0]
    payload = prompt_component.payload

    try:
        if payload.provider == KindModelProvider.LITELLM:
            model = litellm_factory(
                name=payload.model,
                endpoint=payload.model_endpoint,
                api_key=payload.model_api_key,
            )

            completion = await model.generate(
                messages=payload.content,
                stream=chat_request.stream,
            )
        else:
            completion = await _generate_completion(
                anthropic_claude_factory, payload, stream=chat_request.stream
            )

        if isinstance(completion, AsyncIterator):
            return await _handle_stream(completion)
        return ChatResponse(
            response=completion.text,
            metadata=ChatResponseMetadata(
                provider=payload.provider,
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
