from time import time
from typing import Annotated, AsyncIterator, Optional

import structlog
from dependency_injector.providers import Factory
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Request, status

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import X_GITLAB_LANGUAGE_SERVER_VERSION
from ai_gateway.api.snowplow_context import get_snowplow_code_suggestion_context
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
from ai_gateway.async_dependency_resolver import get_container_application
from ai_gateway.cloud_connector import CloudConnectorConfig
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsChunk,
    LanguageServerVersion,
    ModelProvider,
)
from ai_gateway.container import ContainerApplication
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.models import KindModelProvider
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.tracking import SnowplowEventContext

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter()


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


@router.post("/completions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
async def completions(
    request: Request,
    payload: CompletionRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    if not current_user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=[CloudConnectorConfig().service_name],
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code suggestions",
        )

    language_server_version = LanguageServerVersion.from_string(
        request.headers.get(X_GITLAB_LANGUAGE_SERVER_VERSION, None)
    )
    component = payload.prompt_components[0]
    code_context = [
        component.payload.content
        for component in payload.prompt_components
        if component.type == CodeEditorComponents.CONTEXT
        and language_server_version.supports_advanced_context()
    ] or None

    snowplow_code_suggestion_context = get_snowplow_code_suggestion_context(
        req=request,
        prefix=component.payload.content_above_cursor,
        suffix=component.payload.content_below_cursor,
        language=component.payload.language_identifier,
        global_user_id=current_user.global_user_id,
    )
    if component.type == CodeEditorComponents.COMPLETION:
        return await code_completion(
            payload=component.payload,
            code_context=code_context,
            snowplow_event_context=snowplow_code_suggestion_context,
        )
    if component.type == CodeEditorComponents.GENERATION:
        return await code_generation(
            current_user=current_user,
            payload=component.payload,
            code_context=code_context,
            prompt_registry=prompt_registry,
            snowplow_event_context=snowplow_code_suggestion_context,
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
    snowplow_event_context: Optional[SnowplowEventContext] = None,
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
        snowplow_event_context=snowplow_event_context,
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
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    generations_vertex_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.vertex.provider
    ],
    generations_anthropic_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.anthropic_default.provider
    ],
    agent_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.agent_factory.provider
    ],
    code_context: list[CodeContextPayload] = None,
    snowplow_event_context: Optional[SnowplowEventContext] = None,
):
    if payload.prompt_id:
        prompt = prompt_registry.get_on_behalf(current_user, payload.prompt_id)
        engine = agent_factory(model__prompt=prompt)
    else:
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
        snowplow_event_context=snowplow_event_context,
        prompt_enhancer=payload.prompt_enhancer,
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
