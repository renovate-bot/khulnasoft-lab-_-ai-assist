from typing import Optional, Union
from time import time

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


# most of the fields created for backward compatibility
class RequestSuggestions(BaseModel):
    model: str = "fastertransformer"
    prompt: Optional[str]
    suffix: Optional[str]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool]
    logprobs: Optional[int] = None
    echo: Optional[bool]
    stop: Optional[Union[str, list]]
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 1
    best_of: Optional[int] = 1
    logit_bias: Optional[dict]
    user: Optional[str]


class ResponseSuggestions(BaseModel):
    # created for backward compatibility
    class Usage(BaseModel):
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int

    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    id: str
    model: str = "codegen"
    object: str = 'text_completion'
    created: int
    choices: list[Choice]
    usage: Optional[Usage]


@router.post("/", response_model=ResponseSuggestions)
@inject
async def completions(
        req: RequestSuggestions,
        code_suggestions: CodeSuggestionsUseCase = Depends(Provide[CodeSuggestionsContainer.usecase]),
):
    suggestion = code_suggestions(req.prompt)

    return ResponseSuggestions(
        id="id",
        created=int(time()),
        choices=[
            ResponseSuggestions.Choice(text=suggestion),
        ],
    )
