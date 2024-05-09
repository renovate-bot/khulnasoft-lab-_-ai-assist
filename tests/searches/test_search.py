from unittest.mock import Mock, patch

import pytest
from google.cloud import discoveryengine as discoveryengine

from ai_gateway.searches.search import VertexAISearch


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


@pytest.mark.parametrize(
    "gl_version, expected_data_store_id",
    [
        ("1.2.3", "gitlab-docs-1-2"),
        ("10.11.12-pre", "gitlab-docs-10-11"),
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

    with patch("ai_gateway.searches.search.MessageToDict") as return_mock:
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


def test_invalid_version(
    mock_search_service_client,
):
    project = "test-project"
    query = "test query"
    gl_version = "invalid version"
    vertex_search = VertexAISearch(mock_search_service_client, project)

    with pytest.raises(ValueError):
        vertex_search.search(query, gl_version)

    mock_search_service_client.serving_config_path.assert_not_called
    mock_search_service_client.search.assert_not_called
