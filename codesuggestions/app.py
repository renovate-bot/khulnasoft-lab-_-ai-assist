import logging
import uvicorn

from logging.config import dictConfig
from dotenv import load_dotenv

from codesuggestions import Config
from codesuggestions.api import create_fast_api_server
from codesuggestions.deps import FastApiContainer, CodeSuggestionsContainer, _PROBS_ENDPOINTS

from codesuggestions.structured_logging import setup_logging
from codesuggestions.profiling import setup_profiling

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server

# load env variables from .env if exists
load_dotenv()

# prepare configuration settings
config = Config()

# configure logging
dictConfig(config.fastapi.uvicorn_logger)


def main():
    fast_api_container = FastApiContainer()
    fast_api_container.config.auth.from_value(config.auth._asdict())
    fast_api_container.config.fastapi.from_value(config.fastapi._asdict())

    code_suggestions_container = CodeSuggestionsContainer()
    code_suggestions_container.config.triton.from_value(config.triton._asdict())

    app = create_fast_api_server()
    setup_logging(app, json_logs=True, log_level="INFO")
    log = logging.getLogger("uvicorn.error")

    setup_profiling(config.profiling, log)

    @app.on_event("startup")
    def on_server_startup():
        fast_api_container.init_resources()
        code_suggestions_container.init_resources()

        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=False,
            should_instrument_requests_inprogress=False,
            excluded_handlers=_PROBS_ENDPOINTS
        )
        instrumentator.instrument(app)
        # https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/10
        log.info(
            "Metrics HTTP server running on http://%s:%d",
            config.fastapi.metrics_host,
            config.fastapi.metrics_port,
        )
        start_http_server(addr=config.fastapi.metrics_host, port=config.fastapi.metrics_port)

    @app.on_event("shutdown")
    def on_server_shutdown():
        fast_api_container.shutdown_resources()
        code_suggestions_container.shutdown_resources()

    # For now, trust all IPs for proxy headers until https://github.com/encode/uvicorn/pull/1611 is available.
    uvicorn.run(app, host=config.fastapi.api_host, port=config.fastapi.api_port, log_config=config.fastapi.uvicorn_logger, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
