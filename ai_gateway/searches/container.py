from typing import Iterator, Optional

from dependency_injector import containers, providers
from google.cloud import discoveryengine as discoveryengine

from ai_gateway.models import mock

from .search import VertexAISearch

__all__ = ["ContainerSearches"]


def _init_vertex_search_service_client(
    mock_model_responses: bool,
) -> Iterator[Optional[discoveryengine.SearchServiceAsyncClient]]:
    if mock_model_responses:
        yield None
        return

    client = discoveryengine.SearchServiceAsyncClient()
    yield client
    client.transport.close()


class ContainerSearches(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    _mock_selector = providers.Callable(
        lambda mock_model_responses: "mocked" if mock_model_responses else "original",
        config.mock_model_responses,
    )

    grpc_client_vertex = providers.Resource(
        _init_vertex_search_service_client,
        mock_model_responses=config.mock_model_responses,
    )

    vertex_search = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            VertexAISearch,
            client=grpc_client_vertex,
            project=config.vertex_search.project,
            fallback_datastore_version=config.vertex_search.fallback_datastore_version,
        ),
        mocked=providers.Factory(mock.SearchClient),
    )
