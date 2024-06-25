from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from dependency_injector import containers
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.api.middleware import MiddlewareAuthentication, MiddlewareLogRequest
from ai_gateway.auth import User, UserClaims
from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.models.base_text import TextGenModelBase

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def tpl_codegen_dir() -> Path:
    assets_dir = Path(__file__).parent / "ai_gateway" / "_assets"
    tpl_dir = assets_dir / "tpl"
    return tpl_dir / "codegen"


@pytest.fixture
def text_gen_base_model():
    model = Mock(spec=TextGenModelBase)
    model.MAX_MODEL_LEN = 1_000
    model.UPPER_BOUND_MODEL_CHARS = model.MAX_MODEL_LEN * 5
    return model


@pytest.fixture(scope="class")
def stub_auth_provider():
    class StubKeyAuthProvider:
        def authenticate(self, token):
            return None

    return StubKeyAuthProvider()


@pytest.fixture(scope="class")
def test_client(fast_api_router, stub_auth_provider, request):
    middlewares = [
        Middleware(RawContextMiddleware),
        MiddlewareLogRequest(),
        MiddlewareAuthentication(stub_auth_provider, False, None),
    ]
    app = FastAPI(middleware=middlewares)
    app.include_router(fast_api_router)
    client = TestClient(app)

    return client


@pytest.fixture
def mock_client(test_client, stub_auth_provider, auth_user):
    with patch.object(stub_auth_provider, "authenticate", return_value=auth_user):
        yield test_client


@pytest.fixture
def mock_container() -> containers.DeclarativeContainer:
    config = Config()
    container_application = ContainerApplication()
    container_application.config.from_dict(config.model_dump())

    return container_application
