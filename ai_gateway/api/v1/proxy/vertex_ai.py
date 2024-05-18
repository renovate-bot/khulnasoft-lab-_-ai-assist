import structlog
from fastapi import APIRouter, Depends, Request
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_categories
from ai_gateway.async_dependency_resolver import get_vertex_ai_proxy_client
from ai_gateway.gitlab_features import (
    FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    GitLabUnitPrimitive,
)
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import VertexAIProxyClient

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.VERTEX_AI.value}" + "/{path:path}")
@requires(GitLabUnitPrimitive.VERTEX_AI_PROXY)
@feature_categories(FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
async def vertex_ai(
    request: Request,
    vertex_ai_proxy_client: VertexAIProxyClient = Depends(get_vertex_ai_proxy_client),
):
    return await vertex_ai_proxy_client.proxy(request)
