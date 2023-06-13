from pathlib import Path

from dependency_injector import containers, providers

from codesuggestions.auth import GitLabAuthProvider, GitLabOidcProvider
from codesuggestions.api import middleware
from codesuggestions.models import (
    grpc_connect_triton,
    GitLabCodeGen,
    PalmTextGenModel,
    vertex_ai_init,
)
from codesuggestions.suggestions.processing import (
    ModelEngineCodegen,
    ModelEnginePalm,
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


def _init_vertex_ai(project: str, location: str):
    vertex_ai_init(project, location)


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

    telemetry_middleware = providers.Factory(
        middleware.MiddlewareModelTelemetry,
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
        location=config.palm_text_model.location
    )

    model_codegen = providers.Singleton(
        GitLabCodeGen,
        grpc_client=grpc_client,
    )

    engine_codegen = providers.Singleton(
        ModelEngineCodegen.from_local_templates,
        tpl_dir=Path(__file__).parent / "_assets" / "tpl" / "codegen",
        model=model_codegen,
    )

    model_palm = providers.Singleton(
        PalmTextGenModel,
        model_name=config.palm_text_model.name,
    )

    engine_palm = providers.Singleton(
        ModelEnginePalm,
        model=model_palm,
    )

    usecase = providers.Singleton(
        CodeSuggestionsUseCase,
        model=model_codegen,
    )

    usecase_v2 = providers.Singleton(
        CodeSuggestionsUseCaseV2,
        engine_codegen=engine_codegen,
        engine_palm=engine_palm,
    )
