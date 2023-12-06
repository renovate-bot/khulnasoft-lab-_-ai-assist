from typing import Optional

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, validator
from pydantic.types import Json, conlist, constr
from pydantic.typing import Literal
from starlette.authentication import requires

from ai_gateway.deps import XRayContainer
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicModel,
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
    prompt: constr(max_length=400000)
    provider: Literal[AnthropicModel.MODEL_ENGINE]
    model: Literal[AnthropicModel.CLAUDE_INSTANT, AnthropicModel.CLAUDE]


class AnyPromptComponent(BaseModel):
    type: constr(strip_whitespace=True, max_length=255)
    payload: Json
    metadata: Optional[dict[constr(max_length=100), constr(max_length=255)]]

    @validator("metadata")
    def validate_medatada(cls, dictionary):
        if dictionary is not None and len(dictionary) > 10:
            raise ValueError("metadata cannot has more than 10 elements")

        return dictionary


class PackageFilePromptComponent(AnyPromptComponent):
    type: Literal["x_ray_package_file_prompt"] = "x_ray_package_file_prompt"
    payload: PackageFilePromptPayload


class XRayRequest(BaseModel):
    prompt_components: conlist(PackageFilePromptComponent, min_items=1, max_items=1)


@router.post("/libraries")
@requires("code_suggestions")
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
            return completion.text

    except (AnthropicAPIConnectionError, AnthropicAPIStatusError) as ex:
        log.error(f"failed to execute Anthropic request: {ex}")
    return ""
