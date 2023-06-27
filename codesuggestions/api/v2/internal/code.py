from time import time
from typing import Literal, Union, Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, constr
from pydantic.fields import Field
from pydantic.types import confloat, conint

from starlette_context import context

from codesuggestions.api.rollout import ModelRollout

__all__ = [
    "router"
]

log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter(
    prefix="/code",
    tags=["internal"],
)


class ModelGitLabCodegen(BaseModel):
    class Parameters(BaseModel):
        temperature: confloat(ge=0.0, le=1.0) = 0.2
        max_decode_steps: conint(ge=1, le=64) = 32
        top_p: confloat(ge=0.0, le=1.0) = 0.98
        top_k: conint(ge=1, le=40) = 0

    name: Literal[ModelRollout.GITLAB_CODEGEN]
    prefix: constr(max_length=100000)
    parameters: Parameters


class ModelVertexTextBison(BaseModel):
    # Ref: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#text_model_parameters
    class Parameters(BaseModel):
        temperature: confloat(ge=0.0, le=1.0) = 0.2
        max_decode_steps: conint(ge=1, le=1_024) = 16
        top_p: confloat(ge=0.0, le=1.0) = 0.95
        top_k: conint(ge=1, le=40) = 40

    name: Literal[ModelRollout.GOOGLE_TEXT_BISON]
    content: constr(max_length=100000)
    parameters: Parameters


class ModelVertexCodeBison(BaseModel):
    # Ref: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#code-generation-prompt-parameters
    class Parameters(BaseModel):
        temperature: confloat(ge=0.0, le=1.0) = 0.2
        max_decode_steps: conint(ge=1, le=2_048) = 16

    name: Literal[ModelRollout.GOOGLE_CODE_BISON]
    prefix: constr(max_length=100000)
    parameters: Parameters


class ModelVertexCodeGecko(BaseModel):
    # Ref: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#code-completion-prompt-parameters
    class Parameters(BaseModel):
        temperature: confloat(ge=0.0, le=1.0) = 0.2
        max_decode_steps: conint(ge=1, le=64) = 16

    name: Literal[ModelRollout.GOOGLE_CODE_GECKO]
    prefix: constr(max_length=100000)
    suffix: constr(max_length=100000)
    parameters: Parameters


class CodeCompletionsRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    prompt_version: Literal[1] = 1
    project_id: Optional[int]
    project_path: Optional[constr(strip_whitespace=True, max_length=255)]
    file_name: Optional[constr(strip_whitespace=True, max_length=255)]
    model: Union[
        ModelVertexTextBison,
        ModelVertexCodeBison,
        ModelVertexCodeGecko,
        ModelGitLabCodegen,
    ] = Field(..., discriminator="name")


class CodeCompletionsResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    class Model(BaseModel):
        engine: str
        name: str

    id: str
    model: Model
    object: str = "code_completions"
    created: int
    choices: list[Choice]


@router.post("/completions", response_model=CodeCompletionsResponse)
async def completions(
    payload: CodeCompletionsRequest,
):
    return CodeCompletionsResponse(
        id=payload.id,
        created=int(time()),
        model=CodeCompletionsResponse.Model(
            engine=context.get("model_engine", ""), name=context.get("model_name", "")
        ),
        choices=[
            CodeCompletionsResponse.Choice(text=""),
        ],
    )
