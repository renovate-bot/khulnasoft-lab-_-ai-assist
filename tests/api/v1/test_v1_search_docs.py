from time import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.search.typing import (
    DEFAULT_PAGE_SIZE,
    SearchResponse,
    SearchResponseDetails,
    SearchResponseMetadata,
    SearchResult,
)
from ai_gateway.auth import User, UserClaims
from ai_gateway.config import Config


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(scopes=["documentation_search"]),
    )


@pytest.fixture
def request_body():
    return {
        "type": "search-docs",
        "metadata": {"source": "GitLab EE", "version": "17.0.0"},
        "payload": {"query": "What is gitlab mission?"},
    }


@pytest.fixture
def search_results():
    return [
        {
            "id": "doc_id_1",
            "content": "GitLab's mission is to make software development easier and more efficient.",
            "metadata": {
                "source": "GitLab Docs",
                "version": "17.0.0",
                "source_url": "https://docs.gitlab.com/ee/foo",
            },
        }
    ]


@pytest.fixture
def mock_config():
    config = Config()
    config.custom_models.enabled = False

    yield config


@pytest.mark.asyncio
async def test_success(
    mock_client: TestClient,
    request_body: dict,
    search_results: dict,
):

    time_now = time()
    with patch(
        "ai_gateway.searches.search.VertexAISearch.search_with_retry",
        return_value=search_results,
    ) as mock_search_with_retry:
        with patch("time.time", return_value=time_now):
            response = mock_client.post(
                "/search/gitlab-docs",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json=request_body,
            )

    assert response.status_code == 200

    expected_response = SearchResponse(
        response=SearchResponseDetails(
            results=[
                SearchResult(
                    id=result["id"],
                    content=result["content"],
                    metadata=result["metadata"],
                )
                for result in search_results
            ]
        ),
        metadata=SearchResponseMetadata(
            provider="vertex-ai",
            timestamp=int(time_now),
        ),
    )

    assert response.json() == expected_response.dict()

    mock_search_with_retry.assert_called_once_with(
        query=request_body["payload"]["query"],
        gl_version=request_body["metadata"]["version"],
        page_size=DEFAULT_PAGE_SIZE,
    )


@pytest.mark.asyncio
async def test_missing_param(
    mock_client: TestClient,
):
    request_body = {
        "type": "search-docs",
        "metadata": {"source": "GitLab EE", "version": "17.0.0"},
        "payload": {},
    }

    response = mock_client.post(
        "/search/gitlab-docs",
        headers={
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
        },
        json=request_body,
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_authenication(
    mock_client: TestClient,
    request_body: request_body,
):
    response = mock_client.post(
        "/search/gitlab-docs",
        json=request_body,
    )

    assert response.status_code == 401
