import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette_context import context
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.api.middleware import (
    MiddlewareAuthentication,
    MiddlewareLogRequest,
    MiddlewareModelTelemetry,
)
from ai_gateway.api.monitoring import router as http_monitoring_router
from ai_gateway.api.v1 import api_router as http_api_router_v1
from ai_gateway.api.v2 import api_router as http_api_router_v2
from ai_gateway.api.v3 import api_router as http_api_router_v3
from ai_gateway.auth import GitLabOidcProvider
from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.instrumentators.threads import monitor_threads
from ai_gateway.profiling import setup_profiling
from ai_gateway.structured_logging import setup_logging

__all__ = [
    "create_fast_api_server",
]

_SKIP_ENDPOINTS = ["/monitoring/healthz", "/metrics"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.extra["extra"]["config"]
    container_application = ContainerApplication()
    container_application.config.from_dict(config.model_dump())
    container_application.init_resources()

    if config.instrumentator.thread_monitoring_enabled:
        loop = asyncio.get_running_loop()
        loop.create_task(
            monitor_threads(
                loop, interval=config.instrumentator.thread_monitoring_interval
            )
        )

    # https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/10
    log = logging.getLogger("uvicorn.error")
    log.info(
        "Metrics HTTP server running on http://%s:%d",
        config.fastapi.metrics_host,
        config.fastapi.metrics_port,
    )
    start_http_server(
        addr=config.fastapi.metrics_host, port=config.fastapi.metrics_port
    )

    yield

    container_application.shutdown_resources()


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
                GitLabOidcProvider(
                    oidc_providers={
                        "Gitlab": config.gitlab_url,
                        "CustomersDot": config.customer_portal_url,
                    }
                ),
                bypass_auth=config.auth.bypass_external,
                skip_endpoints=_SKIP_ENDPOINTS,
            ),
            MiddlewareModelTelemetry(skip_endpoints=_SKIP_ENDPOINTS),
        ],
        extra={"config": config},
    )

    setup_custom_exception_handlers(fastapi_app)
    setup_router(fastapi_app)
    setup_logging(fastapi_app, config.logging)
    setup_prometheus_fastapi_instrumentator(fastapi_app)
    setup_profiling(config.google_cloud_profiler)

    return fastapi_app


async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    context["http_exception_details"] = str(exc)
    return await http_exception_handler(request, exc)


def setup_custom_exception_handlers(app: FastAPI):
    app.add_exception_handler(StarletteHTTPException, custom_http_exception_handler)


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
