from logging.config import dictConfig

from dotenv import load_dotenv

from ai_gateway.api import create_fast_api_server
from ai_gateway.config import Config

# load env variables from .env if exists
load_dotenv()

# prepare configuration settings
config = Config()

# configure logging
dictConfig(config.fastapi.uvicorn_logger)


def get_config() -> Config:
    return config


def get_app():
    app = create_fast_api_server(config)
    return app
