import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE, feature_categories
from ai_gateway.async_dependency_resolver import (
    get_abuse_detector,
    get_internal_event_client,
    get_vertex_ai_proxy_client,
)
from ai_gateway.auth.request import authorize_with_unit_primitive_header
from ai_gateway.gitlab_features import FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import VertexAIProxyClient

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.VERTEX_AI.value}" + "/{path:path}")
@authorize_with_unit_primitive_header()
@feature_categories(FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
async def vertex_ai(
    request: Request,
    background_tasks: BackgroundTasks,
    abuse_detector: AbuseDetector = Depends(get_abuse_detector),
    vertex_ai_proxy_client: VertexAIProxyClient = Depends(get_vertex_ai_proxy_client),
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    unit_primitive = request.headers[X_GITLAB_UNIT_PRIMITIVE]
    internal_event_client.track_event(
        f"request_{unit_primitive}",
        category=__name__,
    )

    return await vertex_ai_proxy_client.proxy(request)
