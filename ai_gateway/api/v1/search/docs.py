import time
from typing import Annotated

from dependency_injector.providers import Factory
from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory, GitLabUnitPrimitive

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.search.typing import (
    SearchRequest,
    SearchResponse,
    SearchResponseDetails,
    SearchResponseMetadata,
    SearchResult,
)
from ai_gateway.async_dependency_resolver import (
    get_internal_event_client,
    get_search_factory_provider,
)
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.searches import Searcher

__all__ = [
    "router",
]

from ai_gateway.structured_logging import get_request_logger

router = APIRouter()

request_log = get_request_logger("search")


@router.post(
    "/gitlab-docs", response_model=SearchResponse, status_code=status.HTTP_200_OK
)
@feature_category(GitLabFeatureCategory.DUO_CHAT)
async def docs(
    request: Request,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    search_request: SearchRequest,
    search_factory: Factory[Searcher] = Depends(get_search_factory_provider),
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    if not current_user.can(GitLabUnitPrimitive.DOCUMENTATION_SEARCH):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to search documentations",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.DOCUMENTATION_SEARCH}",
        category=__name__,
    )

    payload = search_request.payload

    search_params = {
        "query": payload.query,
        "page_size": payload.page_size,
        "gl_version": search_request.metadata.version,
    }

    searcher = search_factory()

    response = await searcher.search_with_retry(**search_params)

    results = [
        SearchResult(
            id=result["id"], content=result["content"], metadata=result["metadata"]
        )
        for result in response
    ]

    request_log.info(
        "Search completed",
        search_params=search_params,
        results_metadata=[result.metadata for result in results],
    )

    return SearchResponse(
        response=SearchResponseDetails(
            results=results,
        ),
        metadata=SearchResponseMetadata(
            provider=searcher.provider(),
            timestamp=int(time.time()),
        ),
    )
