import json
import os.path
import sqlite3
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException, status
from google.api_core.exceptions import NotFound
from google.cloud import discoveryengine as discoveryengine
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct

from ai_gateway.searches.search import DataStoreNotFound, VertexAISearch


@pytest.fixture
def mock_vertex_search_struct_data():
    return {
        "content": "GitLab's mission is to make software development easier and more efficient.",
        "metadata": {
            "source": "GitLab Docs",
            "version": "17.0.0",
            "source_url": "https://docs.gitlab.com/ee/foo",
        },
    }


@pytest.fixture
def mock_vertex_search_response(mock_vertex_search_struct_data):
    response_dict = {
        "results": [
            {
                "document": {
                    "id": "1",
                    "struct_data": ParseDict(mock_vertex_search_struct_data, Struct()),
                },
            }
        ],
    }

    return discoveryengine.SearchResponse(**response_dict)


@pytest.fixture
def mock_vertex_search_request():
    with patch("ai_gateway.searches.container.discoveryengine.SearchRequest") as mock:
        yield mock


@pytest.fixture
def mock_search_service_client(mock_vertex_search_response):
    with patch(
        "ai_gateway.searches.container.discoveryengine.SearchServiceAsyncClient"
    ) as mock:
        mock.search = AsyncMock(return_value=mock_vertex_search_response)
        mock.serving_config_path.return_value = "path/to/service_config"
        yield mock


@pytest.fixture
def vertex_ai_search_factory():
    def create(
        client: discoveryengine.SearchServiceAsyncClient,
        project: str = "test-project",
        fallback_datastore_version: str = "17.0.0",
    ) -> VertexAISearch:
        return VertexAISearch(
            client=client,
            project=project,
            fallback_datastore_version=fallback_datastore_version,
        )

    return create


@pytest.mark.asyncio
async def test_vertex_ai_search_with_retry_success_first_attempt(
    mock_search_service_client, vertex_ai_search_factory
):
    query = "test query"
    gl_version = "17.1.0"

    vertex_search = vertex_ai_search_factory(
        client=mock_search_service_client, fallback_datastore_version="17.0.0"
    )

    with patch.object(VertexAISearch, "search", return_value=None) as mock_search:
        await vertex_search.search_with_retry(query, gl_version)

        mock_search.assert_called_once_with(query, "17.1.0")


@pytest.mark.asyncio
async def test_vertex_ai_search_with_retry_success_second_attempt(
    mock_search_service_client, vertex_ai_search_factory
):
    query = "test query"
    gl_version = "17.1.0"

    vertex_search = vertex_ai_search_factory(
        client=mock_search_service_client, fallback_datastore_version="17.0.0"
    )

    with patch.object(VertexAISearch, "search", return_value=None) as mock_search:
        mock_search.side_effect = [
            DataStoreNotFound("Data store not found", input="17.1.0"),
            AsyncMock(),
        ]

        await vertex_search.search_with_retry(query, gl_version)

        mock_search.assert_has_calls(
            [call(query, "17.1.0"), call(query, "17.1.0", gl_version="17.0.0")]
        )


@pytest.mark.asyncio
async def test_vertex_ai_search_with_retry_failed_all_attempts(
    mock_search_service_client, vertex_ai_search_factory
):
    query = "test query"
    gl_version = "17.1.0"

    vertex_search = vertex_ai_search_factory(
        client=mock_search_service_client, fallback_datastore_version="17.0.0"
    )

    with pytest.raises(HTTPException):
        with patch.object(VertexAISearch, "search", return_value=None) as mock_search:
            mock_search.side_effect = [
                HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Data store not found.",
                ),
                HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Data store not found.",
                ),
            ]

            await vertex_search.search_with_retry(query, gl_version)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gl_version, expected_data_store_id",
    [
        ("1.2.3", "gitlab-docs-1-2"),
        ("10.11.12-pre", "gitlab-docs-10-11"),
    ],
)
async def test_vertex_ai_search(
    mock_search_service_client,
    mock_vertex_search_request,
    mock_vertex_search_response,
    mock_vertex_search_struct_data,
    gl_version,
    expected_data_store_id,
    vertex_ai_search_factory,
):
    project = "test-project"
    query = "test query"

    vertex_search = vertex_ai_search_factory(
        client=mock_search_service_client, project=project
    )

    result = await vertex_search.search(query, gl_version)
    assert result == [{**mock_vertex_search_struct_data, **{"id": "1"}}]

    mock_search_service_client.serving_config_path.assert_called_once_with(
        project=project,
        location="global",
        data_store=expected_data_store_id,
        serving_config="default_config",
    )
    mock_search_service_client.search.assert_called_once_with(
        mock_vertex_search_request.return_value
    )


@pytest.mark.asyncio
async def test_invalid_version(mock_search_service_client, vertex_ai_search_factory):
    query = "test query"
    gl_version = "invalid version"
    vertex_search = vertex_ai_search_factory(client=mock_search_service_client)

    with pytest.raises(DataStoreNotFound):
        await vertex_search.search(query, gl_version)

    mock_search_service_client.serving_config_path.assert_not_called
    mock_search_service_client.search.assert_not_called


@pytest.mark.asyncio
async def test_datastore_not_found(
    mock_search_service_client,
    vertex_ai_search_factory,
):
    query = "test query"
    gl_version = "15.0.0"

    mock_search_service_client.search.side_effect = NotFound("not found")

    vertex_search = vertex_ai_search_factory(client=mock_search_service_client)

    with pytest.raises(DataStoreNotFound):
        await vertex_search.search(query, gl_version)
