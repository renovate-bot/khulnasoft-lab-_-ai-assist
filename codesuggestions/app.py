import uvicorn
from logging.config import dictConfig
from dotenv import load_dotenv

from codesuggestions import Config
from codesuggestions.api import create_fast_api_server
from codesuggestions.deps import FastApiContainer, CodeSuggestionsContainer

from codesuggestions.structured_logging import setup_logging

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

    @app.on_event("startup")
    def on_server_startup():
        fast_api_container.init_resources()
        code_suggestions_container.init_resources()

    @app.on_event("shutdown")
    def on_server_shutdown():
        fast_api_container.shutdown_resources()
        code_suggestions_container.shutdown_resources()

    uvicorn.run(app, host=config.fastapi.api_host, port=config.fastapi.api_port, log_config=config.fastapi.uvicorn_logger)


if __name__ == "__main__":
    main()
