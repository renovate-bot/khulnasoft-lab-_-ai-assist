from time import time
from typing import Annotated, AsyncIterator, Optional, Tuple, Union

import anthropic
import structlog
from dependency_injector.providers import Factory
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from starlette.datastructures import CommaSeparatedStrings

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import (
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_LANGUAGE_SERVER_VERSION,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER,
    X_GITLAB_SAAS_NAMESPACE_IDS_HEADER,
)
from ai_gateway.api.v2.code.typing import (
    CompletionsRequestV1,
    CompletionsRequestV2,
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
from ai_gateway.auth.self_signed_jwt import SELF_SIGNED_TOKEN_ISSUER
from ai_gateway.auth.user import GitLabUser, get_current_user
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
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.instrumentators.base import TelemetryInstrumentator
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.models import KindAnthropicModel, KindLiteLlmModel, KindModelProvider
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.prompts.typing import ModelMetadata
from ai_gateway.tracking import SnowplowEvent, SnowplowEventContext
from ai_gateway.tracking.errors import log_exception
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator

__all__ = [
    "router",
]


log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter()

CompletionsRequestWithVersion = Annotated[
    Union[CompletionsRequestV1, CompletionsRequestV2],
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
async def completions(
    request: Request,
    payload: CompletionsRequestWithVersion,
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
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
    completions_agent_factory: Factory[CodeCompletions] = Depends(
        get_code_suggestions_completions_agent_factory_provider
    ),
    snowplow_instrumentator: SnowplowInstrumentator = Depends(
        get_snowplow_instrumentator
    ),
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    if not current_user.can(GitLabUnitPrimitive.CODE_SUGGESTIONS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code completions",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.CODE_SUGGESTIONS}",
        category=__name__,
    )

    try:
        snowplow_instrumentator.watch(
            _suggestion_requested_snowplow_event(request, payload)
        )
    except Exception as e:
        log_exception(e)

    log.debug(
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
        code_completions = completions_anthropic_factory()

        # We support the prompt version 2 only with the Anthropic models
        if payload.prompt_version == 2:
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
        and payload.model_name == KindLiteLlmModel.CODESTRAL_2405
    ):
        code_completions = _resolve_code_completions_vertex_codestral(
            payload=payload,
            completions_litellm_factory=completions_litellm_factory,
        )

        # We need to pass this here since litellm.LiteLlmTextGenModel
        # sets the default temperature and max_output_tokens in the `generate` function signature
        # To override those values, the kwargs passed to `generate` is updated here
        # For further details, see:
        #     https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1172#note_2060587592
        #
        # The temperature value is taken from Mistral's docs: https://docs.mistral.ai/api/#operation/createFIMCompletion
        kwargs.update({"temperature": 0.7, "max_output_tokens": 64})
    else:
        code_completions = completions_legacy_factory()
        if payload.choices_count > 0:
            kwargs.update({"candidate_count": payload.choices_count})

        language_server_version = LanguageServerVersion.from_string(
            request.headers.get(X_GITLAB_LANGUAGE_SERVER_VERSION, None)
        )
        if language_server_version.supports_advanced_context() and payload.context:
            kwargs.update({"code_context": [ctx.content for ctx in payload.context]})

    suggestions = await _execute_code_completion(payload, code_completions, **kwargs)

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
async def generations(
    request: Request,
    payload: GenerationsRequestWithVersion,
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
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
    if not current_user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=[SELF_SIGNED_TOKEN_ISSUER],
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code generations",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.CODE_SUGGESTIONS}",
        category=__name__,
    )

    try:
        snowplow_instrumentator.watch(
            _suggestion_requested_snowplow_event(request, payload)
        )
    except Exception as e:
        log_exception(e)

    log.debug(
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
        )

    if isinstance(suggestion, AsyncIterator):
        return await _handle_stream(suggestion)

    log.debug(
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
    current_user: GitLabUser,
    prompt_registry: BasePromptRegistry,
    generations_agent_factory: Factory[CodeGenerations],
) -> CodeGenerations:
    model_metadata = ModelMetadata(
        name=payload.model_name,
        endpoint=payload.model_endpoint,
        api_key=payload.model_api_key,
        provider="custom_openai",
    )

    prompt = prompt_registry.get_on_behalf(
        current_user, payload.prompt_id, model_metadata
    )

    return generations_agent_factory(model__prompt=prompt)


def _resolve_code_completions_litellm(
    payload: SuggestionsRequest,
    current_user: GitLabUser,
    prompt_registry: BasePromptRegistry,
    completions_agent_factory: Factory[CodeCompletions],
    completions_litellm_factory: Factory[CodeCompletions],
) -> CodeCompletions:
    if payload.prompt_version == 2 and not payload.prompt:
        model_metadata = ModelMetadata(
            name=payload.model_name,
            endpoint=payload.model_endpoint,
            api_key=payload.model_api_key,
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
    completions_litellm_factory: Factory[CodeCompletions],
) -> CodeCompletions:
    if payload.prompt_version == 2 and payload.prompt is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot specify a prompt with the given provider and model combination",
        )

    return completions_litellm_factory(
        model__name=payload.model_name,
        model__provider=payload.model_provider,
        model__api_key=payload.model_api_key,
        model__endpoint=payload.model_endpoint,
    )


def _resolve_agent_code_completions(
    model_metadata: ModelMetadata,
    current_user: GitLabUser,
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
        log.debug(
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


def _suggestion_requested_snowplow_event(
    req: Request,
    payload: SuggestionsRequest,
) -> SnowplowEvent:
    language = lang_from_filename(payload.current_file.file_name) or ""
    if language:
        language = language.name.lower()

    # gitlab-rails 16.3+ sends an X-Gitlab-Realm header
    gitlab_realm = req.headers.get(X_GITLAB_REALM_HEADER)
    # older versions don't serve code suggestions, so we read this from the IDE token claim
    if not gitlab_realm and req.user and req.user.claims:
        gitlab_realm = req.user.claims.gitlab_realm

    is_direct_connection = False
    if (
        req.user
        and req.user.claims
        and req.user.claims.issuer == SELF_SIGNED_TOKEN_ISSUER
    ):
        is_direct_connection = True

    return SnowplowEvent(
        context=SnowplowEventContext(
            prefix_length=len(payload.current_file.content_above_cursor),
            suffix_length=len(payload.current_file.content_below_cursor),
            language=language,
            user_agent=req.headers.get("User-Agent", ""),
            gitlab_realm=gitlab_realm if gitlab_realm else "",
            is_direct_connection=is_direct_connection,
            gitlab_instance_id=req.headers.get(X_GITLAB_INSTANCE_ID_HEADER, ""),
            gitlab_global_user_id=req.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER, ""),
            gitlab_host_name=req.headers.get(X_GITLAB_HOST_NAME_HEADER, ""),
            gitlab_saas_duo_pro_namespace_ids=list(
                CommaSeparatedStrings(
                    req.headers.get(X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER, "")
                )
            ),
            gitlab_saas_namespace_ids=list(
                CommaSeparatedStrings(
                    req.headers.get(X_GITLAB_SAAS_NAMESPACE_IDS_HEADER, "")
                )
            ),
        )
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


async def _execute_code_completion(
    payload: CompletionsRequestWithVersion,
    code_completions: Factory[CodeCompletions | CodeCompletionsLegacy],
    **kwargs: dict,
) -> any:
    with TelemetryInstrumentator().watch(payload.telemetry):
        output = await code_completions.execute(
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            file_name=payload.current_file.file_name,
            editor_lang=payload.current_file.language_identifier,
            stream=payload.stream,
            **kwargs,
        )

    if isinstance(code_completions, CodeCompletions):
        return [output]
    return output
