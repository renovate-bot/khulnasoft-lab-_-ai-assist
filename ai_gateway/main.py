import logging
import os

import uvicorn
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    multiprocess,
    start_http_server,
)

from ai_gateway.app import Config, get_config
from ai_gateway.structured_logging import setup_logging


def start_metrics_server(config: Config):
    log = logging.getLogger("main")
    log.info(
        "Metrics HTTP server running on http://%s:%d",
        config.fastapi.metrics_host,
        config.fastapi.metrics_port,
    )

    registry = REGISTRY

    # pylint: disable=direct-environment-variable-reference
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    # pylint: enable=direct-environment-variable-reference

    start_http_server(
        addr=config.fastapi.metrics_host,
        port=config.fastapi.metrics_port,
        registry=registry,
    )


def run_app():
    config = get_config()

    setup_logging(config.logging)
    start_metrics_server(config)

    # For now, trust all IPs for proxy headers until https://github.com/encode/uvicorn/pull/1611 is available.
    uvicorn.run(
        "ai_gateway.app:get_app",
        host=config.fastapi.api_host,
        port=config.fastapi.api_port,
        log_config=config.fastapi.uvicorn_logger,
        forwarded_allow_ips="*",
        reload=config.fastapi.reload,
        factory=True,
    )


if __name__ == "__main__":
    run_app()
