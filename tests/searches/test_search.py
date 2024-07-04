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

from ai_gateway.searches.search import DataStoreNotFound, SqliteSearch, VertexAISearch


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


@pytest.fixture
def mock_sqlite_search_struct_data():
    return {
        "id": "tmp/gitlab-master-doc/doc/topics/git/lfs/index.md",
        "metadata": {
            "Header1": "Git Large File Storage (LFS)",
            "Header2": "Add a file with Git LFS",
            "Header3": "Add a file type to Git LFS",
            "filename": "tmp/gitlab-master-doc/doc/topics/git/lfs/index.md",
        },
    }


@pytest.fixture
def mock_sqlite_search_response():
    return [
        json.dumps(
            {
                "Header1": "Tutorial: Set up issue boards for team hand-off",
                "filename": "tmp/gitlab-master-doc/doc/foo/index.md",
            }
        ),
        "GitLab's mission is to make software development easier and more efficient.",
    ]


@pytest.fixture
def sqlite_search_factory():
    def create() -> SqliteSearch:
        return SqliteSearch()

    return create


@pytest.fixture
def mock_os_path_to_db():
    current_dir = os.path.dirname(__file__)
    local_docs_example_path = current_dir.replace(
        "searches", "_assets/tpl/tools/searches/local_docs_example.db"
    )
    with patch("posixpath.join", return_value=local_docs_example_path) as mock:
        yield mock


@pytest.fixture
def mock_sqlite_connection(mock_sqlite_search_response):
    with patch("sqlite3.connect") as mock:
        mock.return_value.cursor.return_value.execute.return_value = [
            mock_sqlite_search_response
        ]
        yield mock


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
@pytest.mark.parametrize(
    "gl_version",
    [
        ("1.2.3"),
        ("10.11.12-pre"),
    ],
)
async def test_sqlite_search(
    mock_sqlite_search_struct_data,
    mock_os_path_to_db,
    gl_version,
    sqlite_search_factory,
):
    query = "What is lfs?"
    page_size = 4

    sqlite_search = sqlite_search_factory()

    result = await sqlite_search.search(query, gl_version, page_size)
    assert result[0]["id"] == mock_sqlite_search_struct_data["id"]
    assert result[0]["metadata"] == mock_sqlite_search_struct_data["metadata"]

    mock_os_path_to_db.assert_called_once_with("tmp/docs.db")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gl_version",
    [
        ("1.2.3"),
        ("10.11.12-pre"),
    ],
)
async def test_sqlite_search_with_no_db(
    gl_version,
    sqlite_search_factory,
):
    query = "What is lfs?"
    page_size = 4

    with patch("os.path.isfile", return_value=False) as mock:
        sqlite_search = sqlite_search_factory()

        result = await sqlite_search.search(query, gl_version, page_size)
        assert result == []


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
