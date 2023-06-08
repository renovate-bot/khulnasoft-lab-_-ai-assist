from time import time

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, constr

from codesuggestions.api.timing import timing
from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.suggestions import CodeSuggestionsUseCaseV2
from codesuggestions.api.middleware import GitLabUser

from starlette.concurrency import run_in_threadpool

__all__ = [
    "router",
]

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
    project_path: constr(strip_whitespace=True, max_length=255)
    project_id: int
    current_file: CurrentFile


class SuggestionsResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    id: str
    model: str = "codegen"
    object: str = "text_completion"
    created: int
    choices: list[Choice]


@router.post("", response_model=SuggestionsResponse)
@inject
async def completions(
    req: Request,
    payload: SuggestionsRequest,
    f_flags: dict = Depends(
        Provide[CodeSuggestionsContainer.config.feature_flags]
    ),
    code_suggestions: CodeSuggestionsUseCaseV2 = Depends(
        Provide[CodeSuggestionsContainer.usecase_v2]
    ),
):
    f_third_party_ai_default = resolve_third_party_ai_default(
        req.user,
        f_flags["is_third_party_ai_default"],
    )
    suggestion = await run_in_threadpool(
        get_suggestions,
        code_suggestions,
        payload,
        f_third_party_ai_default,
    )

    return SuggestionsResponse(
        id="id",
        created=int(time()),
        choices=[
            SuggestionsResponse.Choice(text=suggestion),
        ],
    )


def resolve_third_party_ai_default(user: GitLabUser, f_third_party_ai_default: bool) -> bool:
    if is_debug := user.is_debug:
        return is_debug and f_third_party_ai_default

    if claims := user.claims:
        return claims.is_third_party_ai_default and f_third_party_ai_default

    return False


@timing("get_suggestions_duration_s")
def get_suggestions(
    code_suggestions: CodeSuggestionsUseCaseV2,
    req: SuggestionsRequest,
    f_third_party_ai_default: bool,
):
    return code_suggestions(
        req.current_file.content_above_cursor,
        req.current_file.file_name,
        third_party=f_third_party_ai_default,
    )
