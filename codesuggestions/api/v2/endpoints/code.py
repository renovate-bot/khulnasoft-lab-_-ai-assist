from time import time
from typing import Optional

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from pydantic import BaseModel, conlist, constr

from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.instrumentators.base import Telemetry, TelemetryInstrumentator
from codesuggestions.suggestions import CodeCompletions, CodeGenerations

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
    object: str = "text_completion"
    created: int
    choices: list[Choice]


@router.post("/completions", response_model=SuggestionsResponse)
@router.post("/code/completions", response_model=SuggestionsResponse)
@inject
async def completions(
    payload: SuggestionsRequest,
    code_completions: CodeCompletions = Depends(
        Provide[CodeSuggestionsContainer.code_completions]
    ),
):
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
        choices=[
            SuggestionsResponse.Choice(text=suggestion.text),
        ],
    )


@router.post("/code/generations", response_model=SuggestionsResponse)
@inject
async def generations(
    payload: SuggestionsRequest,
    code_generations: CodeGenerations = Depends(
        Provide[CodeSuggestionsContainer.code_generations]
    ),
):
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
