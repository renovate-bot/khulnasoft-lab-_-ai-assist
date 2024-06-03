from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.x_ray.typing import XRayRequest, XRayResponse
from ai_gateway.async_dependency_resolver import get_x_ray_anthropic_claude
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.models import AnthropicModel
from ai_gateway.self_signed_token.token_authority import SELF_SIGNED_TOKEN_ISSUER

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
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
    model: AnthropicModel = Depends(get_x_ray_anthropic_claude),
):
    if not current_user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=[SELF_SIGNED_TOKEN_ISSUER],
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access X Ray",
        )
    package_file_prompt = payload.prompt_components[0].payload

    completion = await model.generate(
        prefix=package_file_prompt.prompt,
        _suffix="",
    )
    return XRayResponse(response=completion.text)
