import time

import structlog
from dependency_injector.providers import Factory
from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.search.typing import (
    SearchRequest,
    SearchResponse,
    SearchResponseDetails,
    SearchResponseMetadata,
    SearchResult,
)
from ai_gateway.async_dependency_resolver import get_vertex_search_factory_provider
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.searches import DataStoreNotFound, VertexAISearch, VertexAPISearchError
from ai_gateway.tracking import log_exception

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("search")

router = APIRouter()


@router.post(
    "/gitlab-docs", response_model=SearchResponse, status_code=status.HTTP_200_OK
)
@requires(GitLabUnitPrimitive.DOCUMENTATION_SEARCH)
@feature_category(GitLabFeatureCategory.DUO_CHAT)
async def docs(
    request: Request,
    search_request: SearchRequest,
    vertex_search_factory: Factory[VertexAISearch] = Depends(
        get_vertex_search_factory_provider
    ),
):
    payload = search_request.payload

    search_params = {
        "query": payload.query,
        "page_size": payload.page_size,
        "gl_version": search_request.metadata.version,
    }

    searcher = vertex_search_factory()

    try:
        response = await searcher.search_with_retry(**search_params)
    except DataStoreNotFound:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Data store not found."
        )
    except VertexAPISearchError as ex:
        log_exception(ex)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vertex API Search Error.",
        )

    results = []
    if "results" in response:
        for r in response["results"]:
            search_result = SearchResult(
                id=r["document"]["id"],
                content=r["document"]["structData"]["content"],
                metadata=r["document"]["structData"]["metadata"],
            )
            results.append(search_result)

    return SearchResponse(
        response=SearchResponseDetails(
            results=results,
        ),
        metadata=SearchResponseMetadata(
            provider="vertex-ai",
            timestamp=int(time.time()),
        ),
    )
