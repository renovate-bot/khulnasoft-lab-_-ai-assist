from time import time

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from pydantic import BaseModel, constr

from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.suggestions import CodeSuggestionsUseCase

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
    repository_name: constr(strip_whitespace=True, max_length=255)
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
    req: SuggestionsRequest,
    code_suggestions: CodeSuggestionsUseCase = Depends(
        Provide[CodeSuggestionsContainer.usecase]
    ),
):
    suggestion = code_suggestions(req.current_file.content_above_cursor)

    return SuggestionsResponse(
        id="id",
        created=int(time()),
        choices=[
            SuggestionsResponse.Choice(text=suggestion),
        ],
    )
