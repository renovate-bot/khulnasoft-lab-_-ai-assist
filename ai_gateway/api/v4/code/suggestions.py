from time import time
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, Request
from gitlab_cloud_connector import GitLabFeatureCategory
from sse_starlette.sse import EventSourceResponse

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v3.code.completions import code_suggestions as v3_code_suggestions
from ai_gateway.api.v3.code.typing import (
    CompletionRequest,
    ModelMetadata,
    ResponseMetadataBase,
    StreamModelEngine,
)
from ai_gateway.api.v4.code.typing import (
    StreamDelta,
    StreamEvent,
    StreamSSEMessage,
    StreamSuggestionChunk,
)
from ai_gateway.async_dependency_resolver import get_container_application
from ai_gateway.code_suggestions import CodeSuggestionsChunk
from ai_gateway.prompts import BasePromptRegistry

__all__ = [
    "router",
]

router = APIRouter()


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


async def handle_stream_sse(
    stream: AsyncIterator[CodeSuggestionsChunk],
    engine: StreamModelEngine,
) -> EventSourceResponse:
    async def _stream_response_generator():
        def _start_message():
            # To minimize redundancy, we're only sending metadata in the first SSE message.
            return StreamSSEMessage(
                event=StreamEvent.START,
                data={
                    "metadata": ResponseMetadataBase(
                        timestamp=int(time()),
                        model=ModelMetadata(
                            engine=engine.model.metadata.engine,
                            name=engine.model.metadata.name,
                        ),
                    ).model_dump(exclude_none=True)
                },
            ).dump_with_json_data()

        def _content_message(chunk):
            return StreamSSEMessage(
                event=StreamEvent.CONTENT_CHUNK,
                data=StreamSuggestionChunk(
                    choices=[
                        StreamSuggestionChunk.Choice(
                            delta=StreamDelta(content=chunk.text)
                        )
                    ],
                ).model_dump(),
            ).dump_with_json_data()

        def _end_message():
            return StreamSSEMessage(event=StreamEvent.END).dump_with_json_data()

        yield _start_message()

        async for chunk in stream:
            yield _content_message(chunk)

        yield _end_message()

    return EventSourceResponse(_stream_response_generator())


@router.post("/suggestions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
async def suggestions(
    request: Request,
    payload: CompletionRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    return await v3_code_suggestions(
        request=request,
        payload=payload,
        current_user=current_user,
        prompt_registry=prompt_registry,
        stream_handler=handle_stream_sse,
    )
