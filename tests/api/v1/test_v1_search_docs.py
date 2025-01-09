from time import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.search.typing import (
    DEFAULT_PAGE_SIZE,
    SearchResponse,
    SearchResponseDetails,
    SearchResponseMetadata,
    SearchResult,
)
from ai_gateway.config import Config
from ai_gateway.internal_events import InternalEventAdditionalProperties


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return CloudConnectorUser(
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
    mock_track_internal_event,
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

    mock_track_internal_event.assert_called_once_with(
        "request_documentation_search",
        category="ai_gateway.api.v1.search.docs",
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


@pytest.mark.asyncio
async def test_custom_models_enabled_token_limiting(
    mock_client: TestClient,
    request_body: dict,
    mock_config,
):
    mock_config.custom_models.enabled = True
    time_now = time()

    # Simulate a response with two items, exceeding the token limit
    search_results = [
        {
            "id": "1",
            "content": "a " * 4000,
            "metadata": {},
        },  # 4000 words * 1.4 = 5600 tokens
        {
            "id": "2",
            "content": "a " * 2000,
            "metadata": {},
        },  # 2000 words * 1.4 = 2800 tokens
    ]

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
    # Expected Results
    expected_results = [
        {
            "id": "1",
            "content": "a " * 4000,
            "metadata": {},
        }
    ]

    assert response.status_code == 200
    assert response.json()["response"]["results"] == expected_results

    mock_search_with_retry.assert_called_once_with(
        query=request_body["payload"]["query"],
        gl_version=request_body["metadata"]["version"],
        page_size=request_body["payload"].get("page_size", DEFAULT_PAGE_SIZE),
    )

    final_results = response.json()["response"]["results"]
    assert len(final_results) == 1  # Only one result due to token limiting
    assert final_results[0]["id"] == "1"  # Validate the included result is correct
