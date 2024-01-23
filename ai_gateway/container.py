from dependency_injector import containers, providers
from py_grpc_prometheus.prometheus_client_interceptor import PromClientInterceptor

from ai_gateway.api import middleware
from ai_gateway.auth import GitLabOidcProvider
from ai_gateway.chat.container import ContainerChat
from ai_gateway.code_suggestions.container import ContainerCodeSuggestions
from ai_gateway.models.container import ContainerModels
from ai_gateway.tracking.container import ContainerTracking

__all__ = [
    "ContainerApplication",
]

from ai_gateway.x_ray.container import ContainerXRay

_PROBS_ENDPOINTS = ["/monitoring/healthz", "/metrics"]


class ContainerFastApi(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    oidc_provider = providers.Singleton(
        GitLabOidcProvider,
        oidc_providers=providers.Dict(
            {
                "Gitlab": config.gitlab_url,
                "CustomersDot": config.customer_portal_url,
            }
        ),
    )

    auth_middleware = providers.Factory(
        middleware.MiddlewareAuthentication,
        oidc_provider,
        bypass_auth=config.auth.bypass_external,
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


class ContainerApplication(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "ai_gateway.api.v1.x_ray.libraries",
            "ai_gateway.api.v1.chat.agent",
            "ai_gateway.api.v2.code.completions",
            "ai_gateway.api.v3.code.completions",
            "ai_gateway.api.server",
            "ai_gateway.api.monitoring",
            "ai_gateway.async_dependency_resolver",
        ]
    )

    config = providers.Configuration(strict=True)

    interceptor = providers.Resource(
        PromClientInterceptor,
        enable_client_handling_time_histogram=True,
        enable_client_stream_receive_time_histogram=True,
        enable_client_stream_send_time_histogram=True,
    )

    pkg_models = providers.Container(
        ContainerModels,
        config=config,
    )
    code_suggestions = providers.Container(
        ContainerCodeSuggestions,
        models=pkg_models,
        config=config.f.code_suggestions,
    )
    x_ray = providers.Container(
        ContainerXRay,
        models=pkg_models,
    )
    chat = providers.Container(
        ContainerChat,
        models=pkg_models,
    )

    snowplow = providers.Container(ContainerTracking, config=config.snowplow)
    fastapi = providers.Container(
        ContainerFastApi,
        config=config,
    )
