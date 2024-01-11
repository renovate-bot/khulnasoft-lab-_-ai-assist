from typing import Iterator, cast

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from ai_gateway.api import create_fast_api_server
from ai_gateway.config import Config, ConfigAuth
from ai_gateway.container import ContainerApplication

_ROUTES_V1 = [
    ("/v1/chat/agent", ["POST"]),
    ("/v1/x-ray/libraries", ["POST"]),
    # with the `ai` prefix
    ("/ai/v1/chat/agent", ["POST"]),
    ("/ai/v1/x-ray/libraries", ["POST"]),
]

_ROUTES_V2 = [
    ("/v2/code/completions", ["POST"]),
    ("/v2/completions", ["POST"]),  # legacy path
    ("/v2/code/generations", ["POST"]),
    # with the `ai` prefix
    ("/ai/v2/code/completions", ["POST"]),
    ("/ai/v2/completions", ["POST"]),  # legacy path
    ("/ai/v2/code/generations", ["POST"]),
]

_ROUTES_V3 = [
    ("/v3/code/completions", ["POST"]),
    # with the `ai` prefix
    ("/ai/v3/code/completions", ["POST"]),
]


@pytest.fixture(scope="module")
def fastapi_server_app() -> Iterator[FastAPI]:
    # Disable authorization for testing purposes
    config = Config(_env_file=None, auth=ConfigAuth(bypass_external=True))

    fast_api_container = ContainerApplication()
    fast_api_container.config.from_dict(config.model_dump())

    yield create_fast_api_server()


@pytest.mark.parametrize("routes_expected", [_ROUTES_V1, _ROUTES_V2, _ROUTES_V3])
class TestServerRoutes:
    def test_routes_available(
        self,
        fastapi_server_app: FastAPI,
        routes_expected: list,
    ):
        routes_expected = [
            (path, method) for path, methods in routes_expected for method in methods
        ]

        routes_actual = [
            (cast(APIRoute, route).path, method)
            for route in fastapi_server_app.routes
            for method in cast(APIRoute, route).methods
        ]

        assert set(routes_expected).issubset(routes_actual)

    def test_routes_reachable(
        self,
        fastapi_server_app: FastAPI,
        routes_expected: list,
    ):
        client = TestClient(fastapi_server_app)

        routes_expected = [
            (path, method) for path, methods in routes_expected for method in methods
        ]

        for path, method in routes_expected:
            res = client.request(method, path)
            if method == "POST":
                # We're checking the route availability only
                assert res.status_code == 422
            else:
                assert False
