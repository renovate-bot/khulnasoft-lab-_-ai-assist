from unittest.mock import Mock, patch

import pytest
from google.cloud import discoveryengine as discoveryengine
from google.protobuf.json_format import MessageToDict

from ai_gateway.searches.container import (
    VertexAISearch,
    convert_version,
    get_data_store_id,
)


@pytest.fixture
def mock_search_response():
    mock_response = Mock(spec=discoveryengine.SearchResponse)
    mock_response._pb = Mock()
    return mock_response


@pytest.fixture
def mock_search_request():
    with patch("ai_gateway.searches.container.discoveryengine.SearchRequest") as mock:
        yield mock


@pytest.fixture
def mock_search_service_client(mock_search_response):
    with patch(
        "ai_gateway.searches.container.discoveryengine.SearchServiceClient"
    ) as mock:
        mock.search = Mock(return_value=mock_search_response)
        yield mock


# Test the VertexAISearch class


@pytest.mark.parametrize(
    "gl_version, expected_data_store_id",
    [
        ("1.2.3", "gitlab-docs-1-2"),
        ("10.11.12", "gitlab-docs-10-11"),
    ],
)
def test_vertex_ai_search(
    mock_search_service_client,
    mock_search_request,
    mock_search_response,
    gl_version,
    expected_data_store_id,
):
    project = "test-project"
    query = "test query"

    vertex_search = VertexAISearch(mock_search_service_client, project)

    with patch("ai_gateway.searches.container.MessageToDict") as return_mock:
        result = vertex_search.search(query, gl_version)
        return_mock.assert_called_once_with(mock_search_response._pb)
        assert result == return_mock.return_value

    mock_search_service_client.serving_config_path.assert_called_once_with(
        project=project,
        location="global",
        data_store=expected_data_store_id,
        serving_config="default_config",
    )
    mock_search_service_client.search.assert_called_once_with(
        mock_search_request.return_value
    )


# Test the convert_version function


@pytest.mark.parametrize(
    "version, expected_output",
    [
        ("1.2", "1-2"),
        ("invalid_version", None),
    ],
)
def test_convert_version(version, expected_output):
    assert convert_version(version) == expected_output


# Test the get_data_store_id function


@pytest.mark.parametrize(
    "gl_version, expected_data_store_id",
    [
        ("1.2.3", "gitlab-docs-1-2"),
        ("invalid_version", None),
    ],
)
def test_get_data_store_id(gl_version, expected_data_store_id):
    assert get_data_store_id(gl_version) == expected_data_store_id
