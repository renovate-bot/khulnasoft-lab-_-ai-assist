import json

import pytest
from dependency_injector.wiring import inject
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.authentication import requires
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from structlog.testing import capture_logs

from ai_gateway.api.middleware import MiddlewareAuthentication
from ai_gateway.auth import User, UserClaims

router = APIRouter(
    prefix="",
    tags=["something"],
)


@router.post("/")
@requires("code_suggestions")
def homepage(request: Request):
    return JSONResponse(
        status_code=200,
        content={
            "authenticated": request.user.is_authenticated,
            "is_debug": request.user.is_debug,
            "scopes": request.auth.scopes,
        },
    )


@pytest.fixture(scope="class")
def fast_api_router():
    return router


expected_log_keys = [
    "url",
    "path",
    "status_code",
    "method",
    "correlation_id",
    "http_version",
    "client_ip",
    "client_port",
    "duration_s",
    "cpu_s",
    "user_agent",
    "event",
    "log_level",
    "gitlab_realm",
    "gitlab_instance_id",
    "gitlab_global_user_id",
]

invalid_authentication_token_type_error = {
    "error": "Invalid authentication token type - only OIDC is supported"
}


@pytest.mark.parametrize(
    (
        "headers",
        "data",
        "expected_status_code",
        "auth_user",
        "expected_response",
        "log_keys",
    ),
    [
        (
            None,
            None,
            401,
            User(
                authenticated=True,
                claims=UserClaims(
                    is_third_party_ai_default=False, scopes=["code_suggestions"]
                ),
            ),
            {"error": "No authorization header presented"},
            [],
        ),
        (
            {"Authorization": "invalid"},
            None,
            401,
            User(
                authenticated=True,
                claims=UserClaims(
                    is_third_party_ai_default=False, scopes=["code_suggestions"]
                ),
            ),
            {"error": "Invalid authorization header"},
            [],
        ),
        (
            {"Authorization": "Bearer 12345"},
            None,
            401,
            User(
                authenticated=True,
                claims=UserClaims(
                    is_third_party_ai_default=False, scopes=["code_suggestions"]
                ),
            ),
            invalid_authentication_token_type_error,
            ["auth_duration_s"],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            None,
            200,
            User(
                authenticated=True,
                claims=UserClaims(
                    is_third_party_ai_default=False, scopes=["code_suggestions"]
                ),
            ),
            {"authenticated": True, "is_debug": False, "scopes": ["code_suggestions"]},
            ["auth_duration_s"],
        ),
        (
            # Invalid scope
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            None,
            403,
            User(
                authenticated=True,
                claims=UserClaims(
                    is_third_party_ai_default=False, scopes=["unsupported_scope"]
                ),
            ),
            {"detail": "Forbidden"},
            ["auth_duration_s"],
        ),
        (
            # Unauthorized user
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            None,
            401,
            User(
                authenticated=False,
                claims=UserClaims(
                    is_third_party_ai_default=False, scopes=["code_suggestions"]
                ),
            ),
            {"error": "Forbidden by auth provider"},
            ["auth_duration_s"],
        ),
    ],
)
def test_failed_authorization_logging(
    mock_client, headers, data, expected_status_code, expected_response, log_keys
):
    with capture_logs() as cap_logs:
        response = mock_client.post("/", headers=headers, data=data)

        assert response.status_code == expected_status_code
        assert response.json() == expected_response

        assert len(cap_logs) == 1
        assert cap_logs[0]["status_code"] == expected_status_code
        assert cap_logs[0]["method"] == "POST"
        assert set(cap_logs[0].keys()) == set(expected_log_keys + log_keys)


def test_bypass_auth(fast_api_router, stub_auth_provider):
    middlewares = [
        MiddlewareAuthentication(stub_auth_provider, True, None),
    ]
    app = FastAPI(middleware=middlewares)
    app.include_router(fast_api_router)
    mock_client = TestClient(app)

    response = mock_client.post("/")

    assert response.json() == {
        "authenticated": True,
        "is_debug": True,
        "scopes": ["code_suggestions"],
    }
