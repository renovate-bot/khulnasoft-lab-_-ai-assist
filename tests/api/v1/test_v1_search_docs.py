from time import time
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.search.typing import (
    SearchResponse,
    SearchResponseDetails,
    SearchResponseMetadata,
    SearchResult,
)
from ai_gateway.auth import User, UserClaims
from ai_gateway.container import ContainerApplication
from ai_gateway.searches.search import VertexAISearch


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
    return {
        "results": [
            {
                "document": {
                    "id": "doc_id_1",
                    "structData": {
                        "content": "GitLab's mission is to make software development easier and more efficient.",
                        "metadata": {
                            "source": "GitLab Docs",
                            "version": "17.0.0",
                            "source_url": "https://docs.gitlab.com/ee/foo",
                        },
                    },
                }
            },
            {
                "document": {
                    "id": "doc_id_2",
                    "structData": {
                        "content": "GitLab's mission is to provide a single application for the entire DevOps lifecycle.",
                        "metadata": {
                            "source": "GitLab Docs",
                            "version": "17.0.0",
                            "source_url": "https://docs.gitlab.com/ee/bar",
                        },
                    },
                }
            },
        ]
    }


@pytest.mark.asyncio
async def test_success(
    mock_client: TestClient,
    request_body: dict,
    search_results: dict,
):
    mock_llm_model = mock.Mock(spec=VertexAISearch)
    mock_llm_model.search = mock.AsyncMock(return_value=search_results)

    container = ContainerApplication()
    with container.searches.vertex_search.override(mock_llm_model):
        response = mock_client.post(
            "/search/gitlab-docs",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json=request_body,
        )

    assert response.status_code == 200

    doc1 = search_results["results"][0]["document"]
    doc2 = search_results["results"][1]["document"]

    expected_response = SearchResponse(
        response=SearchResponseDetails(
            results=[
                SearchResult(
                    id=doc1["id"],
                    content=doc1["structData"]["content"],
                    metadata={
                        "source": doc1["structData"]["metadata"]["source"],
                        "version": doc1["structData"]["metadata"]["version"],
                        "source_url": doc1["structData"]["metadata"]["source_url"],
                    },
                ),
                SearchResult(
                    id=doc2["id"],
                    content=doc2["structData"]["content"],
                    metadata={
                        "source": doc2["structData"]["metadata"]["source"],
                        "version": doc2["structData"]["metadata"]["version"],
                        "source_url": doc2["structData"]["metadata"]["source_url"],
                    },
                ),
            ]
        ),
        metadata=SearchResponseMetadata(
            provider="vertex-ai",
            timestamp=int(time()),
        ),
    )

    assert response.json() == expected_response.dict()

    mock_llm_model.search.assert_called_once_with(
        query=request_body["payload"]["query"],
        gl_version=request_body["metadata"]["version"],
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
