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
from codesuggestions.api.rollout.model import (
    ModelRolloutWithFallbackPlan, ModelRollout,
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


def _init_vertex_grpc_client(api_endpoint: str, real_or_fake):
    if real_or_fake == "fake":
        yield None
        return

    client = grpc_connect_vertex({
        "api_endpoint": api_endpoint,
    })
    yield client
    client.transport.close()


def _create_gitlab_codegen_model_provider(grpc_client_triton, real_or_fake):
    return (
        providers.Selector(
            real_or_fake,
            real=providers.Singleton(
                GitLabCodeGen,
                grpc_client=grpc_client_triton,
            ),
            fake=providers.Singleton(FakeGitLabCodeGenModel),
        )
    )


def _create_palm_codegen_model_providers(grpc_client_vertex, project, location, real_or_fake):
    model_names = [
        ModelRollout.GOOGLE_TEXT_BISON,
        ModelRollout.GOOGLE_CODE_BISON,
        ModelRollout.GOOGLE_CODE_GECKO
    ]

    return {
        name: providers.Selector(
            real_or_fake,
            real=providers.Singleton(
                PalmCodeGenModel.from_model_name,
                client=grpc_client_vertex,
                project=project,
                location=location,
                name=name,
            ),
            fake=providers.Singleton(FakePalmTextGenModel),
        )
        for name in model_names
    }


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
        real_or_fake=config.palm_text_model.real_or_fake,
    )

    palm_model_rollout = providers.Callable(
        # take the first model only as the primary one if several passed
        lambda model_names: ModelRollout(model_names[0]),
        model_names=config.palm_text_model.names,
    )

    model_rollout_plan = providers.Resource(
        ModelRolloutWithFallbackPlan,
        rollout_percentage=config.feature_flags.third_party_rollout_percentage,
        primary_model=palm_model_rollout,
        fallback_model=ModelRollout.GITLAB_CODEGEN,
    )

    model_gitlab_codegen = _create_gitlab_codegen_model_provider(
        grpc_client_triton,
        config.gitlab_codegen_model.real_or_fake,
    )

    models_palm_codegen = _create_palm_codegen_model_providers(
        grpc_client_vertex,
        config.palm_text_model.project,
        config.palm_text_model.location,
        config.palm_text_model.real_or_fake
    )

    engine_codegen_factory_template = providers.Callable(
        ModelEngineCodegen.from_local_templates,
        tpl_dir=Path(__file__).parent / "_assets" / "tpl" / "codegen",
    )

    engine_factory = providers.FactoryAggregate(**{
        ModelRollout.GITLAB_CODEGEN: providers.Singleton(
            engine_codegen_factory_template,
            model=model_gitlab_codegen,
        ),
        **{
            ModelRollout(name): providers.Singleton(
                ModelEnginePalm,
                model=model,
            )
            for name, model in models_palm_codegen.items()
        },
    })

    # Deprecated
    usecase = providers.Factory(
        CodeSuggestionsUseCase,
        model=model_gitlab_codegen,
    )

    usecase_v2 = providers.Factory(
        CodeSuggestionsUseCaseV2,
    )
