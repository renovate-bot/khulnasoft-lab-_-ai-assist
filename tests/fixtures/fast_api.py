from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.api.middleware import MiddlewareAuthentication, MiddlewareLogRequest
from ai_gateway.auth import User, UserClaims

pytestmark = pytest.mark.usefixtures("test_client", "stub_auth_provider", "auth_user")


class StubKeyAuthProvider:
    def authenticate(self, token):
        return None


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(is_third_party_ai_default=False, scopes=["code_suggestions"]),
    )


@pytest.fixture(scope="class")
def stub_auth_provider():
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
