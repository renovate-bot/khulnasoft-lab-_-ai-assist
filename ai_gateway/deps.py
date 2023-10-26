from dependency_injector import containers, providers
from py_grpc_prometheus.prometheus_client_interceptor import PromClientInterceptor

from ai_gateway.api import middleware
from ai_gateway.api.rollout.model import ModelRollout
from ai_gateway.auth import GitLabOidcProvider
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
)
from ai_gateway.code_suggestions.processing import ModelEngineCompletions
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessor as PostProcessorCompletions,
)
from ai_gateway.code_suggestions.processing.pre import TokenizerTokenStrategy
from ai_gateway.experimentation import experiment_registry_provider
from ai_gateway.models import (
    AnthropicModel,
    FakePalmTextGenModel,
    PalmCodeGenModel,
    connect_anthropic,
    grpc_connect_vertex,
)
from ai_gateway.tokenizer import init_tokenizer
from ai_gateway.tracking import (
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

_VERTEX_MODELS_VERSIONS = {
    ModelRollout.GOOGLE_TEXT_BISON: f"{ModelRollout.GOOGLE_TEXT_BISON}@001",
    ModelRollout.GOOGLE_CODE_BISON: f"{ModelRollout.GOOGLE_CODE_BISON}@latest",
    ModelRollout.GOOGLE_CODE_GECKO: f"{ModelRollout.GOOGLE_CODE_GECKO}@latest",
}


_ANTHROPIC_MODELS_VERSIONS = {
    AnthropicModel.CLAUDE: "claude-2",
    AnthropicModel.CLAUDE_INSTANT: "claude-instant-1",
}

_ANTHROPIC_MODELS_OPTS = {
    AnthropicModel.CLAUDE: {},
    AnthropicModel.CLAUDE_INSTANT: {"max_tokens_to_sample": 128},
}


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


def _create_anthropic_model(name, client_anthropic, real_or_fake, **kwargs):
    return providers.Selector(
        real_or_fake,
        real=providers.Singleton(
            AnthropicModel.from_model_name,
            client=client_anthropic,
            name=name,
            **kwargs,
        ),
        # TODO: We need to update our fake models to be generic
        fake=providers.Singleton(FakePalmTextGenModel),
    )


def _create_engine_code_completions(model_provider, tokenizer, experiment_registry):
    return providers.Factory(
        ModelEngineCompletions,
        model=model_provider,
        tokenizer=tokenizer,
        post_processor=providers.Factory(PostProcessorCompletions).provider,
        experiment_registry=experiment_registry,
    )


def _all_vertex_models(
    models_key_name, grpc_client_vertex, project, location, real_or_fake
):
    return {
        model_key: _create_vertex_model(
            model_name,
            grpc_client_vertex,
            project,
            location,
            real_or_fake,
        )
        for model_key, model_name in models_key_name.items()
    }


def _all_anthropic_models(
    models_key_name,
    models_opts,
    client_anthropic,
    real_or_fake,
):
    return {
        model_key: _create_anthropic_model(
            model_name, client_anthropic, real_or_fake, **model_opts
        )
        for (model_key, model_name), model_opts, in zip(
            models_key_name.items(), models_opts.values()
        )
    }


def _all_engines(models, tokenizer):
    experiment_registry = experiment_registry_provider()
    # TODO: add experiment_registry to _create_engine_code_generations
    return {
        ModelRollout.GOOGLE_CODE_GECKO: _create_engine_code_completions(
            models[ModelRollout.GOOGLE_CODE_GECKO],
            tokenizer,
            experiment_registry,
        ),
    }


class FastApiContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["ai_gateway.api.server"])

    config = providers.Configuration()

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
            "ai_gateway.api.v2.endpoints.code",
            "ai_gateway.api.v2.experimental.code",
            "ai_gateway.api.monitoring",
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

    client_anthropic = providers.Resource(connect_anthropic)

    tokenizer = providers.Resource(init_tokenizer)

    models_vertex = _all_vertex_models(
        _VERTEX_MODELS_VERSIONS,
        grpc_client_vertex,
        config.palm_text_model.project,
        config.palm_text_model.location,
        config.palm_text_model.real_or_fake,
    )

    models_anthropic = _all_anthropic_models(
        _ANTHROPIC_MODELS_VERSIONS,
        {
            model_key: {**model_opts, "stop_sequences": ["</new_code>", "Human:"]}
            for model_key, model_opts, in _ANTHROPIC_MODELS_OPTS.items()
        },
        client_anthropic,
        # TODO: We need to update our fake model settings to be generic
        config.palm_text_model.real_or_fake,
    )

    engines = _all_engines(models_vertex, tokenizer)

    # TODO: We keep engine factory to support experimental API endpoints.
    # TODO: Would be great to move such dependencies to a separate experimental container
    engine_factory = providers.FactoryAggregate(**engines)

    code_completions_legacy = providers.Factory(
        CodeCompletionsLegacy, engine=engines[ModelRollout.GOOGLE_CODE_GECKO]
    )

    code_completions_anthropic = providers.Factory(
        CodeCompletions,
        model=models_anthropic[AnthropicModel.CLAUDE_INSTANT],
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )

    code_generations_vertex = providers.Factory(
        CodeGenerations,
        model=models_vertex[ModelRollout.GOOGLE_CODE_BISON],
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )

    code_generations_anthropic = providers.Factory(
        CodeGenerations,
        model=models_anthropic[AnthropicModel.CLAUDE],
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
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
