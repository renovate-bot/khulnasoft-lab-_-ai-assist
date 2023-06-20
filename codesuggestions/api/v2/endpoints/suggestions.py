from time import time
from typing import Optional

import structlog
from dependency_injector.wiring import Provide, inject
from dependency_injector.providers import FactoryAggregate, Factory
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, constr

from codesuggestions.api.timing import timing
from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.suggestions import CodeSuggestionsUseCaseV2
from codesuggestions.api.rollout import ModelRolloutPlan

from starlette.concurrency import run_in_threadpool
from starlette_context import context

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter(
    prefix="/completions",
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


class SuggestionsResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    class Model(BaseModel):
        engine: str
        name: str

    id: str
    model: Model
    object: str = "text_completion"
    created: int
    choices: list[Choice]


@router.post("", response_model=SuggestionsResponse)
@inject
async def completions(
    req: Request,
    payload: SuggestionsRequest,
    model_rollout_plan: ModelRolloutPlan = Depends(
        Provide[CodeSuggestionsContainer.model_rollout_plan]
    ),
    engine_factory: FactoryAggregate = Depends(
        Provide[CodeSuggestionsContainer.engine_factory.provider]
    ),
    code_suggestions: Factory[CodeSuggestionsUseCaseV2] = Depends(
        Provide[CodeSuggestionsContainer.usecase_v2.provider]
    ),
):
    model_name = model_rollout_plan.route(req.user, payload.project_id)
    usecase = code_suggestions(engine=engine_factory(model_name))

    suggestion = await run_in_threadpool(
        get_suggestions,
        usecase,
        payload,
    )

    return SuggestionsResponse(
        id="id",
        created=int(time()),
        model=SuggestionsResponse.Model(
            engine=context.get("model_engine", ""), name=context.get("model_name", "")
        ),
        choices=[
            SuggestionsResponse.Choice(text=suggestion),
        ],
    )


@timing("get_suggestions_duration_s")
def get_suggestions(
    usecase: CodeSuggestionsUseCaseV2,
    req: SuggestionsRequest,
):
    return usecase(
        req.current_file.content_above_cursor,
        req.current_file.file_name,
    )
