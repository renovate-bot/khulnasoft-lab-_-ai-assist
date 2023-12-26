import logging
from logging.config import dictConfig

import uvicorn
from dotenv import load_dotenv
from prometheus_client import start_http_server
from prometheus_fastapi_instrumentator import Instrumentator, metrics

from ai_gateway.api import create_fast_api_server
from ai_gateway.config import Config
from ai_gateway.container import _PROBS_ENDPOINTS, ContainerApplication
from ai_gateway.profiling import setup_profiling
from ai_gateway.structured_logging import setup_logging

# load env variables from .env if exists
load_dotenv()

# prepare configuration settings
config = Config()

# configure logging
dictConfig(config.fastapi.uvicorn_logger)


def main():
    code_suggestions_container = ContainerApplication()
    code_suggestions_container.config.from_dict(config.model_dump())

    app = create_fast_api_server()
    setup_logging(app, config.logging)
    log = logging.getLogger("uvicorn.error")

    setup_profiling(config.google_cloud_profiler, log)

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=False,
        excluded_handlers=_PROBS_ENDPOINTS,
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
        code_suggestions_container.init_resources()

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
        code_suggestions_container.shutdown_resources()

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
