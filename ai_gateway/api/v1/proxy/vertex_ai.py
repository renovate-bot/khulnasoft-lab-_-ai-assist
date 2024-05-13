import structlog
from fastapi import APIRouter, Depends, Request
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.async_dependency_resolver import get_vertex_ai_proxy_client
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import VertexAIProxyClient

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.VERTEX_AI.value}" + "/{path:path}")
@requires("vertex_ai_proxy")
@feature_category("ai_abstraction_layer")
async def vertex_ai(
    request: Request,
    vertex_ai_proxy_client: VertexAIProxyClient = Depends(get_vertex_ai_proxy_client),
):
    return await vertex_ai_proxy_client.proxy(request)
