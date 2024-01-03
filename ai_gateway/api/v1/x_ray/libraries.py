import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Request
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.x_ray.typing import XRayRequest, XRayResponse
from ai_gateway.deps import XRayContainer
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicModel,
)
from ai_gateway.tracking.errors import log_exception

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("x-ray")

router = APIRouter()


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
        log_exception(ex)
    return XRayResponse(response="")
