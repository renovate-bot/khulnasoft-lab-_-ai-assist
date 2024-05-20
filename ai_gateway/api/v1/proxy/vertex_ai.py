from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE, feature_categories
from ai_gateway.async_dependency_resolver import get_vertex_ai_proxy_client
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.gitlab_features import FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import VertexAIProxyClient

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.VERTEX_AI.value}" + "/{path:path}")
@feature_categories(FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
async def vertex_ai(
    request: Request,
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
    vertex_ai_proxy_client: VertexAIProxyClient = Depends(get_vertex_ai_proxy_client),
):
    unit_primitive = request.headers.get(X_GITLAB_UNIT_PRIMITIVE)

    if not current_user.can(unit_primitive):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unauthorized to access {unit_primitive}",
        )

    return await vertex_ai_proxy_client.proxy(request)
