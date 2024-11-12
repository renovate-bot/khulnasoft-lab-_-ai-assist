from time import time
from typing import Annotated, AsyncIterator, Optional, Tuple, Union

import anthropic
from dependency_injector.providers import Factory
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.error_utils import capture_validation_errors
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import X_GITLAB_LANGUAGE_SERVER_VERSION
from ai_gateway.api.snowplow_context import get_snowplow_code_suggestion_context
from ai_gateway.api.v2.code.typing import (
    CompletionsRequestV1,
    CompletionsRequestV2,
    CompletionsRequestV3,
    GenerationsRequestV1,
    GenerationsRequestV2,
    GenerationsRequestV3,
    StreamSuggestionsResponse,
    SuggestionsRequest,
    SuggestionsResponse,
)
from ai_gateway.async_dependency_resolver import (
    get_code_suggestions_completions_agent_factory_provider,
    get_code_suggestions_completions_anthropic_provider,
    get_code_suggestions_completions_litellm_factory_provider,
    get_code_suggestions_completions_litellm_vertex_codestral_factory_provider,
    get_code_suggestions_completions_vertex_legacy_provider,
    get_code_suggestions_generations_agent_factory_provider,
    get_code_suggestions_generations_anthropic_chat_factory_provider,
    get_code_suggestions_generations_anthropic_factory_provider,
    get_code_suggestions_generations_litellm_factory_provider,
    get_code_suggestions_generations_vertex_provider,
    get_container_application,
    get_internal_event_client,
    get_snowplow_instrumentator,
)
from ai_gateway.cloud_connector import (
    CloudConnectorConfig,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
)
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsChunk,
)
from ai_gateway.code_suggestions.base import CodeSuggestionsOutput
from ai_gateway.code_suggestions.language_server import LanguageServerVersion
from ai_gateway.code_suggestions.processing.base import ModelEngineOutput
from ai_gateway.code_suggestions.processing.ops import lang_from_filename
from ai_gateway.config import Config
from ai_gateway.instrumentators.base import TelemetryInstrumentator
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.models import KindAnthropicModel, KindModelProvider
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.models.vertex_text import KindVertexTextModel
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.prompts.typing import ModelMetadata
from ai_gateway.structured_logging import get_request_logger
from ai_gateway.tracking import SnowplowEvent, SnowplowEventContext
from ai_gateway.tracking.errors import log_exception
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator

__all__ = [
    "router",
]


request_log = get_request_logger("codesuggestions")

router = APIRouter()

CompletionsRequestWithVersion = Annotated[
    Union[CompletionsRequestV1, CompletionsRequestV2, CompletionsRequestV3],
    Body(discriminator="prompt_version"),
]

GenerationsRequestWithVersion = Annotated[
    Union[GenerationsRequestV1, GenerationsRequestV2, GenerationsRequestV3],
    Body(discriminator="prompt_version"),
]

COMPLETIONS_AGENT_ID = "code_suggestions/completions"
GENERATIONS_AGENT_ID = "code_suggestions/generations"


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


@router.post("/completions")
@router.post("/code/completions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
@capture_validation_errors()
async def completions(
    request: Request,
    payload: CompletionsRequestWithVersion,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    completions_legacy_factory: Factory[CodeCompletionsLegacy] = Depends(
        get_code_suggestions_completions_vertex_legacy_provider
    ),
    completions_anthropic_factory: Factory[CodeCompletions] = Depends(
        get_code_suggestions_completions_anthropic_provider
    ),
    completions_litellm_factory: Factory[CodeCompletions] = Depends(
        get_code_suggestions_completions_litellm_factory_provider
    ),
    completions_litellm_vertex_codestral_factory: Factory[CodeCompletions] = Depends(
        get_code_suggestions_completions_litellm_vertex_codestral_factory_provider
    ),
    completions_agent_factory: Factory[CodeCompletions] = Depends(
        get_code_suggestions_completions_agent_factory_provider
    ),
    snowplow_instrumentator: SnowplowInstrumentator = Depends(
        get_snowplow_instrumentator
    ),
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    # pylint: disable=too-many-branches
    # We need to check both COMPLETE_CODE and CODE_SUGGESTIONS permissions temporarily
    # since existing JWTs were issued with only CODE_SUGGESTIONS permission.
    # TODO: Remove CODE_SUGGESTIONS permission check once all JWTs have been reissued
    # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/710
    code_suggestions_allowed = current_user.can(
        GitLabUnitPrimitive.COMPLETE_CODE,
    ) or current_user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
    )
    if not code_suggestions_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code completions",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.COMPLETE_CODE}",
        category=__name__,
    )

    snowplow_event_context = None
    try:
        language = lang_from_filename(payload.current_file.file_name)
        language_name = language.name if language else ""
        snowplow_event_context = get_snowplow_code_suggestion_context(
            req=request,
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            language=language_name,
            global_user_id=current_user.global_user_id,
        )
        snowplow_instrumentator.watch(SnowplowEvent(context=snowplow_event_context))
    except Exception as e:
        log_exception(e)

    request_log.info(
        "code completion input:",
        model_name=payload.model_name,
        model_provider=payload.model_provider,
        prompt=payload.prompt if hasattr(payload, "prompt") else None,
        prefix=payload.current_file.content_above_cursor,
        suffix=payload.current_file.content_below_cursor,
        current_file_name=payload.current_file.file_name,
        stream=payload.stream,
    )

    kwargs = {}

    if payload.model_provider == KindModelProvider.ANTHROPIC:
        code_completions = completions_anthropic_factory(
            model__name=payload.model_name,
        )

        # We support the prompt version 3 only with the Anthropic models
        if payload.prompt_version == 3:
            kwargs.update({"raw_prompt": payload.prompt})
    elif payload.model_provider in (
        KindModelProvider.LITELLM,
        KindModelProvider.MISTRALAI,
    ):
        code_completions = _resolve_code_completions_litellm(
            payload=payload,
            current_user=current_user,
            prompt_registry=prompt_registry,
            completions_agent_factory=completions_agent_factory,
            completions_litellm_factory=completions_litellm_factory,
        )

        if payload.context:
            kwargs.update({"code_context": [ctx.content for ctx in payload.context]})
    elif (
        payload.model_provider == KindModelProvider.VERTEX_AI
        and payload.model_name == KindVertexTextModel.CODESTRAL_2405
        # Codestral is currently not supported in asia-* locations
        # This is a temporary change to allow rollout of Codestral to all internal users
        # while we are looking into getting support for Codestral in Asia
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/635
        and _allow_vertex_codestral()
    ):
        code_completions = _resolve_code_completions_vertex_codestral(
            payload=payload,
            completions_litellm_vertex_codestral_factory=completions_litellm_vertex_codestral_factory,
        )

        # We need to pass this here since litellm.LiteLlmTextGenModel
        # sets the default temperature and max_output_tokens in the `generate` function signature
        # To override those values, the kwargs passed to `generate` is updated here
        # For further details, see:
        #     https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1172#note_2060587592
        #
        # The temperature value is taken from Mistral's docs: https://docs.mistral.ai/api/#operation/createFIMCompletion
        # context_max_percent is set to 0.3 to limit the amount of context right now because latency increases with larger context
        kwargs.update(
            {"temperature": 0.7, "max_output_tokens": 64, "context_max_percent": 0.3}
        )
        if payload.context:
            kwargs.update({"code_context": [ctx.content for ctx in payload.context]})
    elif payload.model_provider == KindModelProvider.FIREWORKS:
        kwargs.update({"max_output_tokens": 64, "context_max_percent": 0.3})

        if payload.context:
            kwargs.update({"code_context": [ctx.content for ctx in payload.context]})

        code_completions = _resolve_code_completions_litellm(
            payload=payload,
            current_user=current_user,
            prompt_registry=prompt_registry,
            completions_agent_factory=completions_agent_factory,
            completions_litellm_factory=completions_litellm_factory,
        )
    else:
        code_completions = completions_legacy_factory()
        if payload.choices_count > 0:
            kwargs.update({"candidate_count": payload.choices_count})

        language_server_version = LanguageServerVersion.from_string(
            request.headers.get(X_GITLAB_LANGUAGE_SERVER_VERSION, None)
        )
        if language_server_version.supports_advanced_context() and payload.context:
            kwargs.update({"code_context": [ctx.content for ctx in payload.context]})

    suggestions = await _execute_code_completion(
        payload=payload,
        code_completions=code_completions,
        snowplow_event_context=snowplow_event_context,
        **kwargs,
    )

    if isinstance(suggestions[0], AsyncIterator):
        return await _handle_stream(suggestions[0])
    choices, tokens_consumption_metadata = _completion_suggestion_choices(suggestions)
    return SuggestionsResponse(
        id="id",
        created=int(time()),
        model=SuggestionsResponse.Model(
            engine=suggestions[0].model.engine,
            name=suggestions[0].model.name,
            lang=suggestions[0].lang,
            tokens_consumption_metadata=tokens_consumption_metadata,
        ),
        experiments=suggestions[0].metadata.experiments,
        choices=choices,
    )


@router.post("/code/generations")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
@capture_validation_errors()
async def generations(
    request: Request,
    payload: GenerationsRequestWithVersion,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    generations_vertex_factory: Factory[CodeGenerations] = Depends(
        get_code_suggestions_generations_vertex_provider
    ),
    generations_anthropic_factory: Factory[CodeGenerations] = Depends(
        get_code_suggestions_generations_anthropic_factory_provider
    ),
    generations_anthropic_chat_factory: Factory[CodeGenerations] = Depends(
        get_code_suggestions_generations_anthropic_chat_factory_provider
    ),
    generations_litellm_factory: Factory[CodeGenerations] = Depends(
        get_code_suggestions_generations_litellm_factory_provider
    ),
    generations_agent_factory: Factory[CodeGenerations] = Depends(
        get_code_suggestions_generations_agent_factory_provider
    ),
    snowplow_instrumentator: SnowplowInstrumentator = Depends(
        get_snowplow_instrumentator
    ),
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    # We need to check both GENERATE_CODE and CODE_SUGGESTIONS permissions temporarily
    # since existing JWTs were issued with only CODE_SUGGESTIONS permission.
    # TODO: Remove CODE_SUGGESTIONS permission check once all JWTs have been reissued
    # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/710
    code_suggestions_allowed = current_user.can(
        GitLabUnitPrimitive.GENERATE_CODE,
        disallowed_issuers=[CloudConnectorConfig().service_name],
    ) or current_user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=[CloudConnectorConfig().service_name],
    )
    if not code_suggestions_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code generations",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.GENERATE_CODE}",
        category=__name__,
    )

    snowplow_event_context = None
    try:
        language = lang_from_filename(payload.current_file.file_name)
        language_name = language.name if language else ""
        snowplow_event_context = get_snowplow_code_suggestion_context(
            req=request,
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            language=language_name,
            global_user_id=current_user.global_user_id,
        )
        snowplow_instrumentator.watch(SnowplowEvent(context=snowplow_event_context))
    except Exception as e:
        log_exception(e)

    request_log.debug(
        "code creation input:",
        prompt=payload.prompt if hasattr(payload, "prompt") else None,
        prefix=payload.current_file.content_above_cursor,
        suffix=payload.current_file.content_below_cursor,
        current_file_name=payload.current_file.file_name,
        stream=payload.stream,
        endpoint=payload.model_endpoint,
        api_key="*" * len(payload.model_api_key) if payload.model_api_key else None,
    )

    if payload.prompt_id:
        code_generations = _resolve_prompt_code_generations(
            payload, current_user, prompt_registry, generations_agent_factory
        )
    elif payload.model_provider == KindModelProvider.ANTHROPIC:
        if payload.prompt_version == 3:
            code_generations = _resolve_code_generations_anthropic_chat(
                payload,
                generations_anthropic_chat_factory,
            )
        else:
            code_generations = _resolve_code_generations_anthropic(
                payload,
                generations_anthropic_factory,
            )
    elif payload.model_provider == KindModelProvider.LITELLM:
        code_generations = generations_litellm_factory(
            model__name=payload.model_name,
            model__endpoint=payload.model_endpoint,
            model__api_key=payload.model_api_key,
        )
    else:
        code_generations = generations_vertex_factory()

    if payload.prompt_version in {2, 3}:
        code_generations.with_prompt_prepared(payload.prompt)

    with TelemetryInstrumentator().watch(payload.telemetry):
        suggestion = await code_generations.execute(
            prefix=payload.current_file.content_above_cursor,
            file_name=payload.current_file.file_name,
            editor_lang=payload.current_file.language_identifier,
            model_provider=payload.model_provider,
            stream=payload.stream,
            snowplow_event_context=snowplow_event_context,
        )

    if isinstance(suggestion, AsyncIterator):
        return await _handle_stream(suggestion)

    request_log.debug(
        "code creation suggestion:",
        suggestion=suggestion.text,
        score=suggestion.score,
        language=suggestion.lang,
    )

    return SuggestionsResponse(
        id="id",
        created=int(time()),
        model=SuggestionsResponse.Model(
            engine=suggestion.model.engine,
            name=suggestion.model.name,
            lang=suggestion.lang,
        ),
        choices=_generation_suggestion_choices(suggestion.text),
    )


def _resolve_code_generations_anthropic(
    payload: SuggestionsRequest,
    generations_anthropic_factory: Factory[CodeGenerations],
) -> CodeGenerations:
    model_name = (
        payload.model_name if payload.model_name else KindAnthropicModel.CLAUDE_2_0
    )

    return generations_anthropic_factory(
        model__name=model_name,
        model__stop_sequences=["</new_code>", anthropic.HUMAN_PROMPT],
    )


def _resolve_code_generations_anthropic_chat(
    payload: SuggestionsRequest,
    generations_anthropic_chat_factory: Factory[CodeGenerations],
) -> CodeGenerations:
    return generations_anthropic_chat_factory(
        model__name=payload.model_name,
        model__stop_sequences=["</new_code>"],
    )


def _resolve_prompt_code_generations(
    payload: SuggestionsRequest,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    generations_agent_factory: Factory[CodeGenerations],
) -> CodeGenerations:
    model_metadata = ModelMetadata(
        name=payload.model_name,
        endpoint=payload.model_endpoint,
        api_key=payload.model_api_key,
        provider="custom_openai",
        identifier=payload.model_identifier,
    )

    prompt = prompt_registry.get_on_behalf(
        current_user, payload.prompt_id, model_metadata
    )

    return generations_agent_factory(model__prompt=prompt)


def _resolve_code_completions_litellm(
    payload: SuggestionsRequest,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    completions_agent_factory: Factory[CodeCompletions],
    completions_litellm_factory: Factory[CodeCompletions],
) -> CodeCompletions:
    if payload.prompt_version == 2 and not payload.prompt:
        model_metadata = ModelMetadata(
            name=payload.model_name,
            endpoint=payload.model_endpoint,
            api_key=payload.model_api_key,
            identifier=payload.model_identifier,
            provider="text-completion-openai",
        )

        return _resolve_agent_code_completions(
            model_metadata=model_metadata,
            current_user=current_user,
            prompt_registry=prompt_registry,
            completions_agent_factory=completions_agent_factory,
        )

    return completions_litellm_factory(
        model__name=payload.model_name,
        model__endpoint=payload.model_endpoint,
        model__api_key=payload.model_api_key,
        model__provider=payload.model_provider,
    )


def _resolve_code_completions_vertex_codestral(
    payload: SuggestionsRequest,
    completions_litellm_vertex_codestral_factory: Factory[CodeCompletions],
) -> CodeCompletions:
    if payload.prompt_version == 2 and payload.prompt is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot specify a prompt with the given provider and model combination",
        )

    return completions_litellm_vertex_codestral_factory()


def _resolve_agent_code_completions(
    model_metadata: ModelMetadata,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    completions_agent_factory: Factory[CodeCompletions],
) -> CodeCompletions:
    prompt = prompt_registry.get_on_behalf(
        current_user, COMPLETIONS_AGENT_ID, model_metadata
    )

    return completions_agent_factory(
        model__prompt=prompt,
    )


def _completion_suggestion_choices(
    suggestions: list,
) -> Tuple[list[SuggestionsResponse.Choice], Optional[TokensConsumptionMetadata]]:
    if len(suggestions) == 0:
        return [], None
    choices: list[SuggestionsResponse.Choice] = []

    choices = []
    tokens_consumption_metadata = None
    for suggestion in suggestions:
        request_log.debug(
            "code completion suggestion:",
            suggestion=suggestion.text,
            score=suggestion.score,
            language=suggestion.lang,
        )
        if not suggestion.text:
            continue

        if tokens_consumption_metadata is None:
            # We take the first metadata from the suggestions since they are all the same
            if isinstance(suggestion, ModelEngineOutput):
                tokens_consumption_metadata = suggestion.tokens_consumption_metadata
            elif isinstance(suggestion, CodeSuggestionsOutput) and suggestion.metadata:
                tokens_consumption_metadata = (
                    suggestion.metadata.tokens_consumption_metadata
                )

        choices.append(
            SuggestionsResponse.Choice(
                text=suggestion.text,
            )
        )
    return choices, tokens_consumption_metadata


def _generation_suggestion_choices(text: str) -> list:
    return [SuggestionsResponse.Choice(text=text)] if text else []


def _allow_vertex_codestral():
    return not _get_gcp_location().startswith("asia-")


def _get_gcp_location():
    return Config().google_cloud_platform.location


async def _handle_stream(
    response: AsyncIterator[CodeSuggestionsChunk],
) -> StreamSuggestionsResponse:
    async def _stream_generator():
        async for result in response:
            yield result.text

    return StreamSuggestionsResponse(
        _stream_generator(), media_type="text/event-stream"
    )


async def _execute_code_completion(
    payload: CompletionsRequestWithVersion,
    code_completions: Factory[CodeCompletions | CodeCompletionsLegacy],
    snowplow_event_context: Optional[SnowplowEventContext] = None,
    **kwargs: dict,
) -> any:
    with TelemetryInstrumentator().watch(payload.telemetry):
        output = await code_completions.execute(
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            file_name=payload.current_file.file_name,
            editor_lang=payload.current_file.language_identifier,
            stream=payload.stream,
            snowplow_event_context=snowplow_event_context,
            **kwargs,
        )

    if isinstance(code_completions, CodeCompletions):
        return [output]
    return output
