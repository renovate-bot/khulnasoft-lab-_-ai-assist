import asyncio
import os
from contextlib import asynccontextmanager

import litellm
from fastapi import APIRouter, FastAPI
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette_context import context
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.agents.instrumentator import AgentInstrumentator
from ai_gateway.api.middleware import (
    MiddlewareAuthentication,
    MiddlewareLogRequest,
    MiddlewareModelTelemetry,
)
from ai_gateway.api.monitoring import router as http_monitoring_router
from ai_gateway.api.v1 import api_router as http_api_router_v1
from ai_gateway.api.v2 import api_router as http_api_router_v2
from ai_gateway.api.v3 import api_router as http_api_router_v3
from ai_gateway.auth import CompositeProvider, GitLabOidcProvider, LocalAuthProvider
from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.instrumentators.threads import monitor_threads
from ai_gateway.models import ModelAPIError
from ai_gateway.profiling import setup_profiling
from ai_gateway.structured_logging import setup_app_logging

__all__ = [
    "create_fast_api_server",
]

_SKIP_ENDPOINTS = ["/monitoring/healthz", "/monitoring/ready", "/metrics"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.extra["extra"]["config"]
    container_application = ContainerApplication()
    container_application.config.from_dict(config.model_dump())

    if config.instrumentator.thread_monitoring_enabled:
        loop = asyncio.get_running_loop()
        loop.create_task(
            monitor_threads(
                loop, interval=config.instrumentator.thread_monitoring_interval
            )
        )

    setup_litellm()

    yield


def create_fast_api_server(config: Config):

    fastapi_app = FastAPI(
        title="GitLab Code Suggestions",
        description="GitLab Code Suggestions API to serve code completion predictions",
        openapi_url=config.fastapi.openapi_url,
        docs_url=config.fastapi.docs_url,
        redoc_url=config.fastapi.redoc_url,
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        lifespan=lifespan,
        middleware=[
            Middleware(RawContextMiddleware),
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["POST"],
                allow_headers=["*"],
            ),
            MiddlewareLogRequest(skip_endpoints=_SKIP_ENDPOINTS),
            MiddlewareAuthentication(
                CompositeProvider(
                    [
                        LocalAuthProvider(self_signed_jwt=config.self_signed_jwt),
                        GitLabOidcProvider(
                            oidc_providers={
                                "Gitlab": config.gitlab_url,
                                "CustomersDot": config.customer_portal_url,
                            }
                        ),
                    ]
                ),
                bypass_auth=config.auth.bypass_external,
                bypass_auth_with_header=config.auth.bypass_external_with_header,
                skip_endpoints=_SKIP_ENDPOINTS,
            ),
            MiddlewareModelTelemetry(skip_endpoints=_SKIP_ENDPOINTS),
        ],
        extra={"config": config},
    )

    setup_custom_exception_handlers(fastapi_app)
    setup_router(fastapi_app)
    setup_app_logging(fastapi_app)
    setup_prometheus_fastapi_instrumentator(fastapi_app)
    setup_profiling(config.google_cloud_profiler)
    setup_gcp_service_account(config)

    return fastapi_app


async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    context["http_exception_details"] = str(exc)
    return await http_exception_handler(request, exc)


async def model_api_exception_handler(request: Request, exc: ModelAPIError):
    wrapped_exception = StarletteHTTPException(
        status_code=503,
        detail="Inference failed",
    )
    return await http_exception_handler(request, wrapped_exception)


def setup_custom_exception_handlers(app: FastAPI):
    app.add_exception_handler(StarletteHTTPException, custom_http_exception_handler)
    app.add_exception_handler(ModelAPIError, model_api_exception_handler)


def setup_litellm():
    litellm.callbacks = [AgentInstrumentator()]


def setup_router(app: FastAPI):
    sub_router = APIRouter()
    sub_router.include_router(http_api_router_v1, prefix="/v1")
    sub_router.include_router(http_api_router_v2, prefix="/v2")
    sub_router.include_router(http_api_router_v3, prefix="/v3")
    sub_router.include_router(http_monitoring_router)

    app.include_router(sub_router)


def setup_prometheus_fastapi_instrumentator(app: FastAPI):
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=False,
        excluded_handlers=_SKIP_ENDPOINTS,
    )
    instrumentator.add(
        metrics.latency(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            should_exclude_streaming_duration=True,
            buckets=(0.5, 1, 2.5, 5, 10, 30, 60),
        )
    )
    instrumentator.instrument(app)


def setup_gcp_service_account(config: Config):
    """
    Inject service account credential from the `AIGW_GOOGLE_CLOUD_PLATFORM__SERVICE_ACCOUNT_JSON_KEY` environment variable.
    This method should only be used for testing purpose such as CI/CD pipelines.
    For production environment, we don't use this method but use Application Default Credentials (ADC) authentication instead.
    """
    if config.google_cloud_platform.service_account_json_key:
        with open("/tmp/gcp-service-account.json", "w") as f:
            f.write(config.google_cloud_platform.service_account_json_key)
            # pylint: disable=direct-environment-variable-reference
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                "/tmp/gcp-service-account.json"
            )
            # pylint: enable=direct-environment-variable-reference
