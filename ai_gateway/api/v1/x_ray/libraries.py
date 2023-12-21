from typing import Annotated, List, Literal, Optional

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, StringConstraints, field_validator
from pydantic.types import Json
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.deps import XRayContainer
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicModel,
    KindAnthropicModel,
    KindModelProvider,
)

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("x-ray")

router = APIRouter(
    prefix="",
    tags=["x-ray"],
)


class PackageFilePromptPayload(BaseModel):
    prompt: Annotated[str, StringConstraints(max_length=400000)]
    provider: Literal[KindModelProvider.ANTHROPIC]
    model: Literal[KindAnthropicModel.CLAUDE_INSTANT_1_2, KindAnthropicModel.CLAUDE_2_0]


class AnyPromptComponent(BaseModel):
    type: Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]
    payload: Json
    metadata: Optional[
        dict[
            Annotated[str, StringConstraints(max_length=100)],
            Annotated[str, StringConstraints(max_length=255)],
        ]
    ] = None

    @field_validator("metadata")
    @classmethod
    def validate_medatada(cls, dictionary):
        if dictionary is not None and len(dictionary) > 10:
            raise ValueError("metadata cannot has more than 10 elements")

        return dictionary


class PackageFilePromptComponent(AnyPromptComponent):
    type: Literal["x_ray_package_file_prompt"] = "x_ray_package_file_prompt"
    payload: PackageFilePromptPayload


class XRayRequest(BaseModel):
    prompt_components: Annotated[
        List[PackageFilePromptComponent], Field(min_length=1, max_length=1)
    ]


class XRayResponse(BaseModel):
    response: str


@router.post("/libraries", response_model=XRayResponse)
@requires("code_suggestions")
@feature_category("code_suggestions")
@inject
async def libraries(
    request: Request,
    payload: XRayRequest,
    model: AnthropicModel = Depends(Provide[XRayContainer.anthropic_model]),
):
    package_file_prompt = payload.prompt_components[0].payload

    try:
        if completion := await model.generate(
            prefix=package_file_prompt.prompt,
            _suffix="",
        ):
            return XRayResponse(response=completion.text)

    except (AnthropicAPIConnectionError, AnthropicAPIStatusError) as ex:
        log.error(f"failed to execute Anthropic request: {ex}")
    return XRayResponse(response="")
