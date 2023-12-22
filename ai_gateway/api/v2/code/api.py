from time import time
from typing import Annotated, AsyncIterator, List, Literal, Optional, Union

import anthropic
import structlog
from dependency_injector.providers import Factory
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
)
from starlette.datastructures import CommaSeparatedStrings

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import (
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_SAAS_NAMESPACE_IDS_HEADER,
)
from ai_gateway.auth.authentication import requires
from ai_gateway.code_suggestions import (
    PROVIDERS_MODELS_MAP,
    USE_CASES_MODELS_MAP,
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsChunk,
    KindUseCase,
)
from ai_gateway.code_suggestions.processing.ops import lang_from_filename
from ai_gateway.deps import CodeSuggestionsContainer
from ai_gateway.experimentation.base import ExperimentTelemetry
from ai_gateway.instrumentators.base import Telemetry, TelemetryInstrumentator
from ai_gateway.models import AnthropicModel, KindAnthropicModel, KindModelProvider
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter(
    prefix="",
    tags=["completions"],
)


class CurrentFile(BaseModel):
    file_name: Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]
    language_identifier: Optional[
        Annotated[str, StringConstraints(max_length=255)]
    ] = None  # https://code.visualstudio.com/docs/languages/identifiers
    content_above_cursor: Annotated[str, StringConstraints(max_length=100000)]
    content_below_cursor: Annotated[str, StringConstraints(max_length=100000)]


class SuggestionsRequest(BaseModel):
    # Opt out protected namespace "model_" (https://github.com/pydantic/pydantic/issues/6322).
    model_config = ConfigDict(protected_namespaces=())

    project_path: Optional[
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]
    ] = None
    project_id: Optional[int] = None
    current_file: CurrentFile
    model_provider: Optional[KindModelProvider] = None
    model_name: Optional[
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=50)]
    ] = None

    telemetry: Annotated[List[Telemetry], Field(max_length=10)] = []
    stream: Optional[bool] = False


class CompletionsRequest(SuggestionsRequest):
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str, info: ValidationInfo) -> str:
        """Validate model name and model provider are compatible."""

        return _validate_model_name(
            value, KindUseCase.CODE_COMPLETIONS, info.data.get("model_provider")
        )


class GenerationsRequest(SuggestionsRequest):
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str, info: ValidationInfo) -> str:
        """Validate model name and model provider are compatible."""

        return _validate_model_name(
            value, KindUseCase.CODE_GENERATIONS, info.data.get("model_provider")
        )


class CompletionsRequestV1(CompletionsRequest):
    prompt_version: Literal[1] = 1


class GenerationsRequestV1(GenerationsRequest):
    prompt_version: Literal[1] = 1


class CompletionsRequestV2(CompletionsRequest):
    prompt_version: Literal[2]
    prompt: str


class GenerationsRequestV2(GenerationsRequest):
    prompt_version: Literal[2]
    prompt: str


CompletionsRequestWithVersion = Annotated[
    Union[CompletionsRequestV1, CompletionsRequestV2],
    Body(discriminator="prompt_version"),
]

GenerationsRequestWithVersion = Annotated[
    Union[GenerationsRequestV1, GenerationsRequestV2],
    Body(discriminator="prompt_version"),
]


class SuggestionsResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    class Model(BaseModel):
        engine: str
        name: str
        lang: str

    id: str
    model: Model
    experiments: list[ExperimentTelemetry] = []
    object: str = "text_completion"
    created: int
    choices: list[Choice]


class StreamSuggestionsResponse(StreamingResponse):
    pass


@router.post("/completions")
@router.post("/code/completions")
@requires("code_suggestions")
@feature_category("code_suggestions")
@inject
async def completions(
    request: Request,
    payload: CompletionsRequestWithVersion,
    code_completions_legacy: Factory[CodeCompletionsLegacy] = Depends(
        Provide[CodeSuggestionsContainer.code_completions_legacy.provider]
    ),
    code_completions_anthropic: Factory[CodeCompletions] = Depends(
        Provide[CodeSuggestionsContainer.code_completions_anthropic.provider]
    ),
    snowplow_instrumentator: SnowplowInstrumentator = Depends(
        Provide[CodeSuggestionsContainer.snowplow_instrumentator]
    ),
):
    try:
        track_snowplow_event(request, payload, snowplow_instrumentator)
    except Exception as e:
        log.error(f"failed to track Snowplow event: {e}")

    log.debug(
        "code completion input:",
        prompt=payload.prompt if hasattr(payload, "prompt") else None,
        prefix=payload.current_file.content_above_cursor,
        suffix=payload.current_file.content_below_cursor,
        current_file_name=payload.current_file.file_name,
        stream=payload.stream,
    )

    kwargs = {}
    if payload.model_provider == KindModelProvider.ANTHROPIC:
        code_completions = code_completions_anthropic()

        # We support the prompt version 2 only with the Anthropic models
        if payload.prompt_version == 2:
            kwargs.update({"raw_prompt": payload.prompt})
    else:
        code_completions = code_completions_legacy()

    with TelemetryInstrumentator().watch(payload.telemetry):
        suggestion = await code_completions.execute(
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            file_name=payload.current_file.file_name,
            editor_lang=payload.current_file.language_identifier,
            stream=payload.stream,
            **kwargs,
        )

    if isinstance(suggestion, AsyncIterator):
        return await _handle_stream(suggestion)

    log.debug(
        "code completion suggestion:",
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
        experiments=suggestion.metadata.experiments,
        choices=_suggestion_choices(suggestion.text),
    )


@router.post("/code/generations")
@requires("code_suggestions")
@feature_category("code_suggestions")
@inject
async def generations(
    request: Request,
    payload: GenerationsRequestWithVersion,
    anthropic_model: Factory[AnthropicModel] = Depends(
        Provide[CodeSuggestionsContainer.anthropic_model.provider]
    ),
    code_generations_vertex: Factory[CodeGenerations] = Depends(
        Provide[CodeSuggestionsContainer.code_generations_vertex.provider]
    ),
    code_generations_anthropic: Factory[CodeGenerations] = Depends(
        Provide[CodeSuggestionsContainer.code_generations_anthropic.provider]
    ),
    snowplow_instrumentator: SnowplowInstrumentator = Depends(
        Provide[CodeSuggestionsContainer.snowplow_instrumentator]
    ),
):
    try:
        track_snowplow_event(request, payload, snowplow_instrumentator)
    except Exception as e:
        log.error(f"failed to track Snowplow event: {e}")

    log.debug(
        "code creation input:",
        prompt=payload.prompt if hasattr(payload, "prompt") else None,
        prefix=payload.current_file.content_above_cursor,
        suffix=payload.current_file.content_below_cursor,
        current_file_name=payload.current_file.file_name,
        stream=payload.stream,
    )

    if payload.model_provider == KindModelProvider.ANTHROPIC:
        code_generations = _resolve_code_generations_anthropic(
            payload=payload,
            anthropic_model=anthropic_model,
            code_generations_anthropic=code_generations_anthropic,
        )
    else:
        code_generations = code_generations_vertex()

    if payload.prompt_version == 2:
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
        choices=_suggestion_choices(suggestion.text),
    )


def _resolve_code_generations_anthropic(
    payload: SuggestionsRequest,
    anthropic_model: Factory[AnthropicModel],
    code_generations_anthropic: Factory[CodeGenerations],
) -> CodeGenerations:
    model_name = (
        payload.model_name
        if payload.model_name
        else KindAnthropicModel.CLAUDE_2_0.value
    )
    anthropic_opts = {
        "model_name": model_name,
        "stop_sequences": ["</new_code>", anthropic.HUMAN_PROMPT],
    }
    model = anthropic_model(**anthropic_opts)

    return code_generations_anthropic(model=model)


def _suggestion_choices(text: str) -> list:
    return [SuggestionsResponse.Choice(text=text)] if text else []


def track_snowplow_event(
    req: Request,
    payload: SuggestionsRequest,
    snowplow_instrumentator: SnowplowInstrumentator,
):
    language = lang_from_filename(payload.current_file.file_name) or ""
    if language:
        language = language.name.lower()

    # gitlab-rails 16.3+ sends an X-Gitlab-Realm header
    gitlab_realm = req.headers.get(X_GITLAB_REALM_HEADER)
    # older versions don't serve code suggestions, so we read this from the IDE token claim
    if not gitlab_realm and req.user and req.user.claims:
        gitlab_realm = req.user.claims.gitlab_realm

    snowplow_instrumentator.watch(
        telemetry=payload.telemetry,
        prefix_length=len(payload.current_file.content_above_cursor),
        suffix_length=len(payload.current_file.content_below_cursor),
        language=language,
        user_agent=req.headers.get("User-Agent", ""),
        gitlab_realm=gitlab_realm if gitlab_realm else "",
        gitlab_instance_id=req.headers.get(X_GITLAB_INSTANCE_ID_HEADER, ""),
        gitlab_global_user_id=req.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER, ""),
        gitlab_host_name=req.headers.get(X_GITLAB_HOST_NAME_HEADER, ""),
        gitlab_saas_namespace_ids=list(
            CommaSeparatedStrings(
                req.headers.get(X_GITLAB_SAAS_NAMESPACE_IDS_HEADER, "")
            )
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


def _validate_model_name(
    model_name: str,
    use_case: KindUseCase,
    provider: Optional[KindModelProvider] = None,
) -> str:
    # ignore model name validation when provider is invalid
    if not provider:
        return model_name

    use_case_models = USE_CASES_MODELS_MAP.get(use_case)
    provider_models = PROVIDERS_MODELS_MAP.get(provider)

    if not use_case_models or not provider_models:
        raise ValueError(f"model {model_name} is unknown")

    valid_model_names = use_case_models & set(provider_models)

    if model_name not in valid_model_names:
        raise ValueError(
            f"model {model_name} is not supported by use case {use_case} and provider {provider}"
        )

    return model_name
