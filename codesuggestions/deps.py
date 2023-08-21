from dependency_injector import containers, providers
from py_grpc_prometheus.prometheus_client_interceptor import PromClientInterceptor

from codesuggestions.api import middleware
from codesuggestions.api.rollout.model import ModelRollout
from codesuggestions.auth import GitLabAuthProvider, GitLabOidcProvider
from codesuggestions.experimentation import experiment_registry_provider
from codesuggestions.models import (
    FakePalmTextGenModel,
    PalmCodeGenModel,
    grpc_connect_vertex,
)
from codesuggestions.suggestions import CodeCompletions, CodeGenerations
from codesuggestions.suggestions.processing import (
    ModelEngineCompletions,
    ModelEngineGenerations,
)
from codesuggestions.tokenizer import init_tokenizer
from codesuggestions.tracking import (
    SnowplowClient,
    SnowplowClientConfiguration,
    SnowplowClientStub,
    SnowplowInstrumentator,
)

__all__ = [
    "FastApiContainer",
    "CodeSuggestionsContainer",
]

_PROBS_ENDPOINTS = ["/monitoring/healthz", "/metrics"]


def _init_vertex_grpc_client(api_endpoint: str, real_or_fake):
    if real_or_fake == "fake":
        yield None
        return

    client = grpc_connect_vertex(
        {
            "api_endpoint": api_endpoint,
        }
    )
    yield client
    client.transport.close()


def _init_snowplow_client(enabled: bool, configuration: SnowplowClientConfiguration):
    if not enabled:
        return SnowplowClientStub()

    return SnowplowClient(configuration)


def _create_vertex_model(name, grpc_client_vertex, project, location, real_or_fake):
    return providers.Selector(
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


def _create_engine_code_completions(model_provider, tokenizer, experiment_registry):
    return providers.Factory(
        ModelEngineCompletions,
        model=model_provider,
        tokenizer=tokenizer,
        experiment_registry=experiment_registry,
    )


def _create_engine_code_generations(model_provider, tokenizer):
    return providers.Factory(
        ModelEngineGenerations,
        model=model_provider,
        tokenizer=tokenizer,
    )


def _all_vertex_models(names, grpc_client_vertex, project, location, real_or_fake):
    return {
        name: _create_vertex_model(
            name,
            grpc_client_vertex,
            project,
            location,
            real_or_fake,
        )
        for name in names
    }


def _all_engines(models, tokenizer):
    experiment_registry = experiment_registry_provider()
    # TODO: add experiment_registry to _create_engine_code_generations
    return {
        ModelRollout.GOOGLE_CODE_GECKO: _create_engine_code_completions(
            models[ModelRollout.GOOGLE_CODE_GECKO], tokenizer, experiment_registry
        ),
        **{
            model_name: _create_engine_code_generations(models[model_name], tokenizer)
            for model_name in [
                ModelRollout.GOOGLE_TEXT_BISON,
                ModelRollout.GOOGLE_CODE_BISON,
            ]
        },
    }


class FastApiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=["codesuggestions.api.server"]
    )

    config = providers.Configuration()

    auth_provider = providers.Singleton(
        GitLabAuthProvider,
        base_url=config.auth.gitlab_api_base_url,
    )

    oidc_provider = providers.Singleton(
        GitLabOidcProvider,
        oidc_providers=providers.Dict(
            {
                "Gitlab": config.auth.gitlab_base_url,
                "CustomersDot": config.auth.customer_portal_base_url,
            }
        ),
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
            "codesuggestions.api.v2.endpoints.code",
            "codesuggestions.api.v2.experimental.code",
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

    grpc_client_vertex = providers.Resource(
        _init_vertex_grpc_client,
        api_endpoint=config.palm_text_model.vertex_api_endpoint,
        real_or_fake=config.palm_text_model.real_or_fake,
    )

    tokenizer = providers.Resource(init_tokenizer)

    models = _all_vertex_models(
        [
            ModelRollout.GOOGLE_TEXT_BISON,
            ModelRollout.GOOGLE_CODE_BISON,
            ModelRollout.GOOGLE_CODE_GECKO,
        ],
        grpc_client_vertex,
        config.palm_text_model.project,
        config.palm_text_model.location,
        config.palm_text_model.real_or_fake,
    )

    engines = _all_engines(models, tokenizer)

    # TODO: We keep engine factory to support experimental API endpoints.
    # TODO: Would be great to move such dependencies to a separate experimental container
    engine_factory = providers.FactoryAggregate(**engines)

    code_completions = providers.Factory(
        CodeCompletions, engine=engines[ModelRollout.GOOGLE_CODE_GECKO]
    )

    code_generations = providers.Factory(
        CodeGenerations, engine=engines[ModelRollout.GOOGLE_CODE_BISON]
    )

    snowplow_config = providers.Resource(
        SnowplowClientConfiguration,
        endpoint=config.tracking.snowplow_endpoint,
    )

    snowplow_client = providers.Resource(
        _init_snowplow_client,
        enabled=config.tracking.snowplow_enabled,
        configuration=snowplow_config,
    )

    snowplow_instrumentator = providers.Resource(
        SnowplowInstrumentator,
        client=snowplow_client,
    )
