from pathlib import Path

from dependency_injector import containers, providers
from py_grpc_prometheus.prometheus_client_interceptor import PromClientInterceptor

from codesuggestions.auth import GitLabAuthProvider, GitLabOidcProvider
from codesuggestions.api import middleware
from codesuggestions.models import (
    grpc_connect_triton,
    grpc_connect_vertex,
    GitLabCodeGen,
    PalmCodeGenModel,
    FakeGitLabCodeGenModel,
    FakePalmTextGenModel,
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


def _init_triton_grpc_client(host: str, port: int, interceptor: PromClientInterceptor):
    client = grpc_connect_triton(host, port, interceptor)
    yield client
    client.close()


def _init_vertex_grpc_client(api_endpoint: str):
    client = grpc_connect_vertex({
        "api_endpoint": api_endpoint,
    })
    yield client
    client.transport.close()


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

    interceptor = providers.Resource(
        PromClientInterceptor,
        enable_client_handling_time_histogram=True,
        enable_client_stream_receive_time_histogram=True,
        enable_client_stream_send_time_histogram=True,
    )

    grpc_client_triton = providers.Resource(
        _init_triton_grpc_client,
        host=config.triton.host,
        port=config.triton.port,
        interceptor=interceptor,
    )

    grpc_client_vertex = providers.Resource(
        _init_vertex_grpc_client,
        api_endpoint=config.palm_text_model.vertex_api_endpoint,
    )

    model_codegen = providers.Selector(
        config.gitlab_codegen_model.real_or_fake,
        real=providers.Singleton(
            GitLabCodeGen,
            grpc_client=grpc_client_triton,
        ),
        fake=providers.Singleton(FakeGitLabCodeGenModel),
    )

    engine_codegen = providers.Singleton(
        ModelEngineCodegen.from_local_templates,
        tpl_dir=Path(__file__).parent / "_assets" / "tpl" / "codegen",
        model=model_codegen,
    )

    model_palm = providers.Selector(
        config.palm_text_model.real_or_fake,
        real=providers.Singleton(
            PalmCodeGenModel.from_model_name,
            name=config.palm_text_model.name,
            client=grpc_client_vertex,
            project=config.palm_text_model.project,
            location=config.palm_text_model.location,
        ),
        fake=providers.Singleton(FakePalmTextGenModel),
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
