from time import time
from typing import Optional

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, conlist, constr

from codesuggestions.api.middleware import (
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_REALM_HEADER,
)
from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.experimentation.base import ExperimentTelemetry
from codesuggestions.instrumentators.base import Telemetry, TelemetryInstrumentator
from codesuggestions.suggestions import CodeCompletions, CodeGenerations
from codesuggestions.suggestions.processing.ops import lang_from_filename
from codesuggestions.tracking.instrumentator import SnowplowInstrumentator

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter(
    prefix="",
    tags=["completions"],
)


class CurrentFile(BaseModel):
    file_name: constr(strip_whitespace=True, max_length=255)
    content_above_cursor: constr(max_length=100000)
    content_below_cursor: constr(max_length=100000)


class SuggestionsRequest(BaseModel):
    prompt_version: int = 1
    project_path: Optional[constr(strip_whitespace=True, max_length=255)]
    project_id: Optional[int]
    current_file: CurrentFile
    telemetry: conlist(Telemetry, max_items=10) = []


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


@router.post("/completions", response_model=SuggestionsResponse)
@router.post("/code/completions", response_model=SuggestionsResponse)
@inject
async def completions(
    req: Request,
    payload: SuggestionsRequest,
    code_completions: CodeCompletions = Depends(
        Provide[CodeSuggestionsContainer.code_completions]
    ),
    snowplow_instrumentator: SnowplowInstrumentator = Depends(
        Provide[CodeSuggestionsContainer.snowplow_instrumentator]
    ),
):
    try:
        track_snowplow_event(req, payload, snowplow_instrumentator)
    except Exception as e:
        log.error(f"failed to track Snowplow event: {e}")

    with TelemetryInstrumentator().watch(payload.telemetry):
        suggestion = await code_completions(
            payload.current_file.content_above_cursor,
            payload.current_file.content_below_cursor,
            payload.current_file.file_name,
        )

    return SuggestionsResponse(
        id="id",
        created=int(time()),
        model=SuggestionsResponse.Model(
            engine=suggestion.model.engine,
            name=suggestion.model.name,
            lang=suggestion.lang(),
        ),
        experiments=suggestion.metadata.experiments,
        choices=[
            SuggestionsResponse.Choice(text=suggestion.text),
        ],
    )


@router.post("/code/generations", response_model=SuggestionsResponse)
@inject
async def generations(
    req: Request,
    payload: SuggestionsRequest,
    code_generations: CodeGenerations = Depends(
        Provide[CodeSuggestionsContainer.code_generations]
    ),
    snowplow_instrumentator: SnowplowInstrumentator = Depends(
        Provide[CodeSuggestionsContainer.snowplow_instrumentator]
    ),
):
    try:
        track_snowplow_event(req, payload, snowplow_instrumentator)
    except Exception as e:
        log.error(f"failed to track Snowplow event: {e}")

    with TelemetryInstrumentator().watch(payload.telemetry):
        suggestion = await code_generations(
            payload.current_file.content_above_cursor,
            payload.current_file.content_below_cursor,
            payload.current_file.file_name,
        )

    return SuggestionsResponse(
        id="id",
        created=int(time()),
        model=SuggestionsResponse.Model(
            engine=suggestion.model.engine,
            name=suggestion.model.name,
            lang=suggestion.lang(),
        ),
        choices=[
            SuggestionsResponse.Choice(text=suggestion.text),
        ],
    )


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
    )
