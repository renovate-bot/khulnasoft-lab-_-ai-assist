import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.feature_category import feature_categories
from ai_gateway.async_dependency_resolver import (
    get_abuse_detector,
    get_anthropic_proxy_client,
)
from ai_gateway.auth.request import authorize_with_unit_primitive_header
from ai_gateway.gitlab_features import FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import AnthropicProxyClient

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.ANTHROPIC.value}" + "/{path:path}")
@authorize_with_unit_primitive_header()
@feature_categories(FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
async def anthropic(
    request: Request,
    background_tasks: BackgroundTasks,
    abuse_detector: AbuseDetector = Depends(get_abuse_detector),
    anthropic_proxy_client: AnthropicProxyClient = Depends(get_anthropic_proxy_client),
):
    return await anthropic_proxy_client.proxy(request)
