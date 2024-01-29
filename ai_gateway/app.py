import asyncio
import logging
from logging.config import dictConfig

import uvicorn
from dotenv import load_dotenv
from fastapi.exception_handlers import http_exception_handler
from prometheus_client import start_http_server
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette_context import context

from ai_gateway.api import create_fast_api_server
from ai_gateway.config import Config
from ai_gateway.container import (
    _METRICS_ENDPOINTS,
    _PROBS_ENDPOINTS,
    ContainerApplication,
)
from ai_gateway.instrumentators.threads import monitor_threads
from ai_gateway.profiling import setup_profiling
from ai_gateway.structured_logging import setup_logging

# load env variables from .env if exists
load_dotenv()

# prepare configuration settings
config = Config()

# configure logging
dictConfig(config.fastapi.uvicorn_logger)


def main():
    container_application = ContainerApplication()
    container_application.config.from_dict(config.model_dump())

    app = create_fast_api_server()
    setup_logging(app, config.logging)
    log = logging.getLogger("uvicorn.error")

    setup_profiling(config.google_cloud_profiler, log)

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=False,
        excluded_handlers=_PROBS_ENDPOINTS + _METRICS_ENDPOINTS,
    )
    instrumentator.add(
        metrics.latency(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            buckets=(0.5, 1, 2.5, 5, 10, 30, 60),
        )
    )
    instrumentator.instrument(app)

    @app.on_event("startup")
    def on_server_startup():
        container_application.init_resources()

        if config.instrumentator.thread_monitoring_enabled:
            loop = asyncio.get_running_loop()
            loop.create_task(
                monitor_threads(
                    loop, interval=config.instrumentator.thread_monitoring_interval
                )
            )

        # https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/10
        log.info(
            "Metrics HTTP server running on http://%s:%d",
            config.fastapi.metrics_host,
            config.fastapi.metrics_port,
        )
        start_http_server(
            addr=config.fastapi.metrics_host, port=config.fastapi.metrics_port
        )

    @app.on_event("shutdown")
    def on_server_shutdown():
        container_application.shutdown_resources()

    @app.exception_handler(StarletteHTTPException)
    async def custom_http_exception_handler(request, exc):
        context["http_exception_details"] = str(exc)
        return await http_exception_handler(request, exc)

    # For now, trust all IPs for proxy headers until https://github.com/encode/uvicorn/pull/1611 is available.
    uvicorn.run(
        app,
        host=config.fastapi.api_host,
        port=config.fastapi.api_port,
        log_config=config.fastapi.uvicorn_logger,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
