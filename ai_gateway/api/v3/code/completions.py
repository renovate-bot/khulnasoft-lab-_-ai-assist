from time import time
from typing import AsyncIterator

import structlog
from anthropic import HUMAN_PROMPT as anthropic_human_prompt
from dependency_injector.providers import Factory
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v3.code.typing import (
    CodeEditorComponents,
    CompletionRequest,
    CompletionResponse,
    EditorContentCompletionPayload,
    EditorContentGenerationPayload,
    ModelMetadata,
    ResponseMetadataBase,
    StreamSuggestionsResponse,
)
from ai_gateway.auth.authentication import requires
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsChunk,
    ModelProvider,
)
from ai_gateway.container import ContainerApplication

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter()


@router.post("/completions")
@requires("code_suggestions")
@feature_category("code_suggestions")
async def completions(
    request: Request,
    payload: CompletionRequest,
):
    component = payload.prompt_components[0]
    if component.type == CodeEditorComponents.COMPLETION:
        return await code_completion(payload=component.payload)
    if component.type == CodeEditorComponents.GENERATION:
        return await code_generation(payload=component.payload)


@inject
async def code_completion(
    payload: EditorContentCompletionPayload,
    code_completions_legacy: Factory[CodeCompletionsLegacy] = Depends(
        Provide[
            ContainerApplication.code_suggestions.completions.vertex_legacy.provider
        ]
    ),
    code_completions_anthropic: Factory[CodeCompletions] = Depends(
        Provide[ContainerApplication.code_suggestions.completions.anthropic.provider]
    ),
):
    if payload.model_provider == ModelProvider.ANTHROPIC:
        engine = code_completions_anthropic()
    else:
        engine = code_completions_legacy()

    suggestion = await engine.execute(
        prefix=payload.content_above_cursor,
        suffix=payload.content_below_cursor,
        file_name=payload.file_name,
        editor_lang=payload.language_identifier,
        stream=payload.stream,
    )

    if isinstance(suggestion, AsyncIterator):
        return await _handle_stream(suggestion)

    return CompletionResponse(
        response=suggestion.text,
        metadata=ResponseMetadataBase(
            timestamp=int(time()),
            model=ModelMetadata(
                engine=suggestion.model.engine,
                name=suggestion.model.name,
                lang=suggestion.lang,
            ),
        ),
    )


@inject
async def code_generation(
    payload: EditorContentGenerationPayload,
    code_generations_vertex: Factory[CodeGenerations] = Depends(
        Provide[ContainerApplication.code_suggestions.generations.vertex.provider]
    ),
    code_generations_anthropic: Factory[CodeGenerations] = Depends(
        Provide[ContainerApplication.code_suggestions.generations.anthropic.provider]
    ),
):
    if payload.model_provider == ModelProvider.ANTHROPIC:
        engine = _resolve_code_generations_anthropic()
    else:
        engine = code_generations_vertex()

    if payload.prompt:
        engine.with_prompt_prepared(payload.prompt)

    suggestion = await engine.execute(
        prefix=payload.content_above_cursor,
        file_name=payload.file_name,
        editor_lang=payload.language_identifier,
        model_provider=payload.model_provider,
        stream=payload.stream,
    )

    if isinstance(suggestion, AsyncIterator):
        return await _handle_stream(suggestion)

    return CompletionResponse(
        response=suggestion.text,
        metadata=ResponseMetadataBase(
            timestamp=int(time()),
            model=ModelMetadata(
                engine=suggestion.model.engine,
                name=suggestion.model.name,
                lang=suggestion.lang,
            ),
        ),
    )


@inject
def _resolve_code_generations_anthropic(
    anthropic_model: Factory[AnthropicModel] = Depends(
        Provide[CodeSuggestionsContainer.anthropic_model.provider]
    ),
    code_generations_anthropic: Factory[CodeGenerations] = Depends(
        Provide[CodeSuggestionsContainer.code_generations_anthropic.provider]
    ),
) -> CodeGenerations:
    anthropic_opts = {
        "model_name": KindAnthropicModel.CLAUDE_2_0.value,
        "stop_sequences": ["</new_code>", anthropic_human_prompt],
    }
    model = anthropic_model(**anthropic_opts)

    return code_generations_anthropic(model=model)


async def _handle_stream(
    response: AsyncIterator[CodeSuggestionsChunk],
) -> StreamSuggestionsResponse:
    async def _stream_generator():
        async for result in response:
            yield result.text

    return StreamSuggestionsResponse(
        _stream_generator(), media_type="text/event-stream"
    )
