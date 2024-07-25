from time import time
from typing import Annotated, AsyncIterator

import structlog
from dependency_injector.providers import Factory
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Request, status

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v3.code.typing import (
    CodeContextPayload,
    CodeEditorComponents,
    CompletionRequest,
    CompletionResponse,
    EditorContentCompletionPayload,
    EditorContentGenerationPayload,
    ModelMetadata,
    ResponseMetadataBase,
    StreamSuggestionsResponse,
)
from ai_gateway.auth.self_signed_jwt import SELF_SIGNED_TOKEN_ISSUER
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsChunk,
    ModelProvider,
)
from ai_gateway.container import ContainerApplication
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.models import KindModelProvider

__all__ = [
    "router",
]


log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter()


@router.post("/completions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
async def completions(
    request: Request,
    payload: CompletionRequest,
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
):
    if not current_user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=[SELF_SIGNED_TOKEN_ISSUER],
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code suggestions",
        )

    component = payload.prompt_components[0]
    code_context = [
        component.payload.content
        for component in payload.prompt_components
        if component.type == CodeEditorComponents.CONTEXT
    ] or None

    if component.type == CodeEditorComponents.COMPLETION:
        return await code_completion(
            payload=component.payload, code_context=code_context
        )
    if component.type == CodeEditorComponents.GENERATION:
        return await code_generation(
            payload=component.payload, code_context=code_context
        )


@inject
async def code_completion(
    payload: EditorContentCompletionPayload,
    completions_legacy_factory: Factory[CodeCompletionsLegacy] = Provide[
        ContainerApplication.code_suggestions.completions.vertex_legacy.provider
    ],
    completions_anthropic_factory: Factory[CodeCompletions] = Provide[
        ContainerApplication.code_suggestions.completions.anthropic.provider
    ],
    code_context: list[CodeContextPayload] = None,
):
    if payload.model_provider == ModelProvider.ANTHROPIC:
        engine = completions_anthropic_factory()
    else:
        engine = completions_legacy_factory()

    kwargs = {}
    if payload.choices_count > 0:
        kwargs.update({"candidate_count": payload.choices_count})

    suggestions = await engine.execute(
        prefix=payload.content_above_cursor,
        suffix=payload.content_below_cursor,
        file_name=payload.file_name,
        editor_lang=payload.language_identifier,
        stream=payload.stream,
        code_context=code_context,
        **kwargs,
    )

    if not isinstance(suggestions, list):
        suggestions = [suggestions]

    if isinstance(suggestions[0], AsyncIterator):
        return await _handle_stream(suggestions[0])

    return CompletionResponse(
        choices=_completion_suggestion_choices(suggestions),
        metadata=ResponseMetadataBase(
            timestamp=int(time()),
            model=ModelMetadata(
                engine=suggestions[0].model.engine,
                name=suggestions[0].model.name,
                lang=suggestions[0].lang,
            ),
        ),
    )


def _completion_suggestion_choices(suggestions: list) -> list:
    if len(suggestions) == 0:
        return []

    choices = []
    for suggestion in suggestions:
        log.debug(
            "code completion suggestion:",
            suggestion=suggestion,
            score=suggestion.score,
            language=suggestion.lang,
        )

        if not suggestion.text:
            continue

        choices.append(CompletionResponse.Choice(text=suggestion.text))

    return choices


@inject
async def code_generation(
    payload: EditorContentGenerationPayload,
    generations_vertex_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.vertex.provider
    ],
    generations_anthropic_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.anthropic_default.provider
    ],
    code_context: list[CodeContextPayload] = None,
):
    if payload.model_provider == KindModelProvider.ANTHROPIC:
        engine = generations_anthropic_factory()
    else:
        engine = generations_vertex_factory()

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

    choices = (
        [CompletionResponse.Choice(text=suggestion.text)] if suggestion.text else []
    )

    return CompletionResponse(
        choices=choices,
        metadata=ResponseMetadataBase(
            timestamp=int(time()),
            model=ModelMetadata(
                engine=suggestion.model.engine,
                name=suggestion.model.name,
                lang=suggestion.lang,
            ),
        ),
    )


async def _handle_stream(
    response: AsyncIterator[CodeSuggestionsChunk],
) -> StreamSuggestionsResponse:
    async def _stream_generator():
        async for result in response:
            yield result.text

    return StreamSuggestionsResponse(
        _stream_generator(), media_type="text/event-stream"
    )
