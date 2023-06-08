from dependency_injector import containers, providers

from codesuggestions.auth import GitLabAuthProvider, GitLabOidcProvider
from codesuggestions.api import middleware
from codesuggestions.models import (
    grpc_connect_triton,
    GitLabCodeGen,
    PalmTextGenModel,
    vertex_ai_init,
)
from codesuggestions.suggestions import (
    CodeSuggestionsUseCase,
    CodeSuggestionsUseCaseV2,
)

__all__ = [
    "FastApiContainer",
    "CodeSuggestionsContainer",
]

_PROBS_ENDPOINTS = [
    "/monitoring/healthz",
    "/metrics"
]


def _init_triton_grpc_client(host: str, port: int):
    client = grpc_connect_triton(host, port)
    yield client
    client.close()


def _init_vertex_ai(project: str, location: str, is_third_party_ai_default: bool):
    if is_third_party_ai_default:
        vertex_ai_init(project, location)


def _init_palm_text_gen_model(model_name: str, is_third_party_ai_default: bool):
    if is_third_party_ai_default:
        return PalmTextGenModel(
            model_name=model_name,
        )

    return None


class FastApiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["codesuggestions.api.server"])

    config = providers.Configuration()

    auth_provider = providers.Singleton(
        GitLabAuthProvider,
        base_url=config.auth.gitlab_api_base_url,
    )

    oidc_provider = providers.Singleton(
        GitLabOidcProvider,
        base_url=config.auth.gitlab_base_url,
    )

    auth_middleware = providers.Factory(
        middleware.MiddlewareAuthentication,
        auth_provider,
        oidc_provider,
        bypass_auth=config.auth.bypass,
        skip_endpoints=_PROBS_ENDPOINTS,
    )

    log_middleware = providers.Factory(
        middleware.MiddlewareLogRequest,
        skip_endpoints=_PROBS_ENDPOINTS,
    )


class CodeSuggestionsContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "codesuggestions.api.suggestions",
            "codesuggestions.api.v2.endpoints.suggestions",
            "codesuggestions.api.monitoring",
        ]
    )

    config = providers.Configuration()

    grpc_client = providers.Resource(
        _init_triton_grpc_client,
        host=config.triton.host,
        port=config.triton.port,
    )

    _ = providers.Resource(
        _init_vertex_ai,
        project=config.palm_text_model.project,
        location=config.palm_text_model.location,
        is_third_party_ai_default=config.feature_flags.is_third_party_ai_default,
    )

    model_codegen = providers.Singleton(
        GitLabCodeGen,
        grpc_client=grpc_client,
    )

    model_palm = providers.Singleton(
        _init_palm_text_gen_model,
        model_name=config.palm_text_model.name,
        is_third_party_ai_default=config.feature_flags.is_third_party_ai_default,
    )

    usecase = providers.Singleton(
        CodeSuggestionsUseCase,
        model=model_codegen,
    )

    usecase_v2 = providers.Singleton(
        CodeSuggestionsUseCaseV2,
        model_codegen=model_codegen,
        model_palm=model_palm,
    )
