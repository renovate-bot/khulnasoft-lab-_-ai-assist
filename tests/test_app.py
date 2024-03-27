from fastapi import FastAPI

from ai_gateway.app import get_app, get_config
from ai_gateway.config import Config, ConfigFastApi


def test_get_config():
    config = get_config()
    assert isinstance(config, Config)
    assert isinstance(config.fastapi, ConfigFastApi)


def test_get_app():
    app = get_app()
    assert isinstance(app, FastAPI)
