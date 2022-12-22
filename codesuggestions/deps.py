from dependency_injector import containers, providers

from codesuggestions.auth import GitLabAuthProvider
from codesuggestions.api import middleware
from codesuggestions.models import grpc_connect_triton, Codegen
from codesuggestions.suggestions import CodeSuggestionsUseCase

__all__ = [
    "FastApiContainer",
    "CodeSuggestionsContainer",
]


def _init_triton_grpc_client(host: str, port: int):
    client = grpc_connect_triton(host, port)
    yield client
    client.close()


class FastApiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["codesuggestions.api.server"])

    config = providers.Configuration()

    auth_provider = providers.Singleton(
        GitLabAuthProvider,
        base_url=config.auth.gitlab_api_base_url,
    )

    auth_middleware = providers.Factory(
        middleware.MiddlewareAuthentication,
        auth_provider,
        bypass_auth=config.auth.bypass,
    )


class CodeSuggestionsContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["codesuggestions.api.suggestions"])

    config = providers.Configuration()

    grpc_client = providers.Resource(
        _init_triton_grpc_client,
        host=config.triton.host,
        port=config.triton.port,
    )

    model_codegen = providers.Singleton(
        Codegen,
        grpc_client=grpc_client,
    )

    usecase = providers.Singleton(
        CodeSuggestionsUseCase,
        model=model_codegen,
    )
