from typing import Iterator, Optional

from dependency_injector import containers, providers
from google.cloud import discoveryengine as discoveryengine

from .search import VertexAISearch

__all__ = ["ContainerSearches"]


def _init_vertex_search_service_client() -> (
    Iterator[Optional[discoveryengine.SearchServiceClient]]
):
    client = discoveryengine.SearchServiceClient()
    yield client
    client.transport.close()


class ContainerSearches(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    grpc_client_vertex = providers.Resource(
        _init_vertex_search_service_client,
    )

    vertex_search = providers.Factory(
        VertexAISearch,
        client=grpc_client_vertex,
        project=config.vertex_search.project,
    )
