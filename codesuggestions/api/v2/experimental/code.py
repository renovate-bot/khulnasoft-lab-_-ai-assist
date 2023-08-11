from time import time
from typing import Literal, Optional, Union
from uuid import uuid4

import structlog
from dependency_injector.providers import FactoryAggregate
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from pydantic import BaseModel, constr
from pydantic.fields import Field
from pydantic.types import confloat, conint, conlist

from codesuggestions.api.rollout import ModelRollout
from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.instrumentators.base import Telemetry, TelemetryInstrumentator
from codesuggestions.suggestions.experimental import CodeCompletionsInternalUseCase

__all__ = ["router"]

log = structlog.stdlib.get_logger("codesuggestions")

router = APIRouter(
    prefix="/code",
    tags=["internal"],
)


class ModelVertexTextBison(BaseModel):
    # Ref: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#text_model_parameters
    class Parameters(BaseModel):
        temperature: confloat(ge=0.0, le=1.0) = 0.2
        max_output_tokens: conint(ge=1, le=1_024) = 16
        top_p: confloat(ge=0.0, le=1.0) = 0.95
        top_k: conint(ge=1, le=40) = 40

    name: Literal[ModelRollout.GOOGLE_TEXT_BISON]
    content: constr(max_length=100000)
    parameters: Parameters


class ModelVertexCodeBison(BaseModel):
    # Ref: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#code-generation-prompt-parameters
    class Parameters(BaseModel):
        temperature: confloat(ge=0.0, le=1.0) = 0.2
        max_output_tokens: conint(ge=1, le=2_048) = 16

    name: Literal[ModelRollout.GOOGLE_CODE_BISON]
    prefix: constr(max_length=100000)
    parameters: Parameters


class ModelVertexCodeGecko(BaseModel):
    # Ref: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#code-completion-prompt-parameters
    class Parameters(BaseModel):
        temperature: confloat(ge=0.0, le=1.0) = 0.2
        max_output_tokens: conint(ge=1, le=64) = 16

    name: Literal[ModelRollout.GOOGLE_CODE_GECKO]
    prefix: constr(max_length=100000)
    suffix: constr(max_length=100000)
    parameters: Parameters


ModelAny = Union[
    ModelVertexTextBison,
    ModelVertexCodeBison,
    ModelVertexCodeGecko,
]


class CodeCompletionsRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    prompt_version: Literal[1] = 1
    project_id: Optional[int]
    project_path: Optional[constr(strip_whitespace=True, max_length=255)]
    file_name: Optional[constr(strip_whitespace=True, max_length=255)]
    model: ModelAny = Field(..., discriminator="name")
    telemetry: conlist(Telemetry, max_items=10) = []


class CodeCompletionsResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str

    class Model(BaseModel):
        engine: str
        name: str
        lang: str

    id: str
    model: Model
    object: str = "code_completions"
    created: int
    choices: list[Choice]


@router.post("/completions", response_model=CodeCompletionsResponse)
@inject
async def completions(
    payload: CodeCompletionsRequest,
    engine_factory: FactoryAggregate = Depends(
        Provide[CodeSuggestionsContainer.engine_factory.provider]
    ),
):
    engine = engine_factory(payload.model.name)
    usecase = CodeCompletionsInternalUseCase(engine)

    with TelemetryInstrumentator().watch(payload.telemetry):
        completion = await usecase(
            _get_requested_prefix(payload.model),
            _get_requested_suffix(payload.model),
            file_name=payload.file_name,
            **payload.model.parameters.dict(),
        )

    return CodeCompletionsResponse(
        id=payload.id,
        created=int(time()),
        model=CodeCompletionsResponse.Model(
            engine=completion.model.engine,
            name=completion.model.name,
            lang=completion.model.lang,
        ),
        choices=[
            CodeCompletionsResponse.Choice(
                text=completion.text,
                finish_reason=completion.finish_reason,
            ),
        ],
    )


def _get_requested_prefix(model: ModelAny) -> str:
    if model.name == ModelRollout.GOOGLE_TEXT_BISON:
        return model.content

    return model.prefix


def _get_requested_suffix(model: ModelAny) -> str:
    suffix = ""
    if model.name == ModelRollout.GOOGLE_CODE_GECKO:
        suffix = model.suffix

    return suffix
