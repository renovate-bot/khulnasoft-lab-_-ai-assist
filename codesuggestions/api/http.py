from fastapi import APIRouter
from pydantic import BaseModel

__all__ = [
    'router',
]

router = APIRouter(
    prefix="/completions",
    tags=["completions"],
)


class RequestCompletions(BaseModel):
    prompt: str


class ResponseCompletions(BaseModel):
    content: str


@router.post("/", response_model=ResponseCompletions)
def completions(req: RequestCompletions):
    return ResponseCompletions(
        content="test",
    )
