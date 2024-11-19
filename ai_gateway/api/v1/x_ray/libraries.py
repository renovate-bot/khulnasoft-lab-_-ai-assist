from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
)

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.x_ray.typing import XRayRequest, XRayResponse
from ai_gateway.async_dependency_resolver import (
    get_internal_event_client,
    get_x_ray_anthropic_claude,
)
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.models import AnthropicModel

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("x-ray")

router = APIRouter()


@router.post("/libraries", response_model=XRayResponse)
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
async def libraries(
    request: Request,
    payload: XRayRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    model: AnthropicModel = Depends(get_x_ray_anthropic_claude),
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    if not current_user.can(
        GitLabUnitPrimitive.GENERATE_CODE,
        disallowed_issuers=[CloudConnectorConfig().service_name],
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access X Ray",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.GENERATE_CODE}",
        category=__name__,
    )

    package_file_prompt = payload.prompt_components[0].payload

    completion = await model.generate(
        prefix=package_file_prompt.prompt,
        _suffix="",
    )
    return XRayResponse(response=completion.text)
