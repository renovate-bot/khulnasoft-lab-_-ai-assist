import structlog
from fastapi import APIRouter, Depends, Request
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.async_dependency_resolver import get_anthropic_proxy_client
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import AnthropicProxyClient

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.ANTHROPIC.value}" + "/{path:path}")
@requires("anthropic_proxy")
@feature_category("ai_abstraction_layer")
async def anthropic(
    request: Request,
    anthropic_proxy_client: AnthropicProxyClient = Depends(get_anthropic_proxy_client),
):
    return await anthropic_proxy_client.proxy(request)
