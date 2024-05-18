import structlog
from fastapi import APIRouter, Depends, Request
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.x_ray.typing import XRayRequest, XRayResponse
from ai_gateway.async_dependency_resolver import get_x_ray_anthropic_claude
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.models import AnthropicModel

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("x-ray")

router = APIRouter()


@router.post("/libraries", response_model=XRayResponse)
@requires(GitLabUnitPrimitive.CODE_SUGGESTIONS)
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
async def libraries(
    request: Request,
    payload: XRayRequest,
    model: AnthropicModel = Depends(get_x_ray_anthropic_claude),
):
    package_file_prompt = payload.prompt_components[0].payload

    completion = await model.generate(
        prefix=package_file_prompt.prompt,
        _suffix="",
    )
    return XRayResponse(response=completion.text)
