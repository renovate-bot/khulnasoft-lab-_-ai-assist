from fastapi import APIRouter, Depends
from pydantic import BaseModel
from dependency_injector.wiring import Provide, inject

from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.suggestions import CodeSuggestionsUseCase

__all__ = [
    "router",
]

router = APIRouter(
    prefix="/completions",
    tags=["completions"],
)


class RequestSuggestions(BaseModel):
    prompt: str


class ResponseSuggestions(BaseModel):
    content: str


@router.post("/", response_model=ResponseSuggestions)
@inject
def completions(
        req: RequestSuggestions,
        code_suggestions: CodeSuggestionsUseCase = Depends(Provide[CodeSuggestionsContainer.usecase]),
):
    suggestions = code_suggestions(req.prompt)

    return ResponseSuggestions(
        content=suggestions,
    )
