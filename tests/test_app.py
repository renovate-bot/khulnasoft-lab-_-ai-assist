from fastapi import FastAPI

from ai_gateway.app import get_app, get_config
from ai_gateway.config import ConfigFastApi


def test_get_config():
    config = get_config()
    assert config is not None
    assert isinstance(config.fastapi, ConfigFastApi)


def test_get_app():
    app = get_app()
    assert isinstance(app, FastAPI)
