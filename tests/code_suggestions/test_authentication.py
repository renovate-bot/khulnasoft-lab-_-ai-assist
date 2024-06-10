import json
from typing import Annotated

import pytest
from dependency_injector.wiring import inject
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from structlog.testing import capture_logs

from ai_gateway.api.middleware import MiddlewareAuthentication
from ai_gateway.auth import User, UserClaims
from ai_gateway.auth.user import GitLabUser, get_current_user

router = APIRouter(
    prefix="",
    tags=["something"],
)


@router.post("/")
def homepage(
    request: Request, current_user: Annotated[GitLabUser, Depends(get_current_user)]
):
    if not (
        (current_user.can("feature1") or current_user.can("feature2"))
        and current_user.can("feature3")
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access homepage",
        )

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
    "gitlab_host_name",
    "gitlab_version",
    "gitlab_saas_namespace_ids",
    "gitlab_saas_duo_pro_namespace_ids",
    "duration_request",
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
                claims=UserClaims(scopes=["feature1", "feature3"]),
            ),
            {"error": "No authorization header presented"},
            ["auth_error_details"],
        ),
        (
            {"Authorization": "invalid"},
            None,
            401,
            User(
                authenticated=True,
                claims=UserClaims(scopes=["feature1", "feature3"]),
            ),
            {"error": "Invalid authorization header"},
            ["auth_error_details"],
        ),
        (
            {"Authorization": "Bearer 12345"},
            None,
            401,
            User(
                authenticated=True,
                claims=UserClaims(scopes=["feature1", "feature3"]),
            ),
            invalid_authentication_token_type_error,
            ["auth_duration_s", "auth_error_details"],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            200,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="1234",
                    gitlab_realm="self-managed",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature3"],
            },
            ["auth_duration_s", "token_issuer"],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            200,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature2", "feature3"],
                    subject="1234",
                    gitlab_realm="self-managed",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature2", "feature3"],
            },
            ["auth_duration_s", "token_issuer"],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            200,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature1", "feature2", "feature3"],
                    subject="1234",
                    gitlab_realm="self-managed",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature2", "feature3"],
            },
            ["auth_duration_s", "token_issuer"],
        ),
        (
            #  No scopes in the token
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            403,
            User(
                authenticated=True,
                claims=UserClaims(subject="1234", gitlab_realm="self-managed"),
            ),
            {"detail": "Unauthorized to access homepage"},
            ["auth_duration_s", "token_issuer"],
        ),
        (
            #  Missing feature3 scope
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            403,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature1"], subject="1234", gitlab_realm="self-managed"
                ),
            ),
            {"detail": "Unauthorized to access homepage"},
            ["auth_duration_s", "token_issuer"],
        ),
        (
            # Missing feature1 or feature2 scopes
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            403,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature3"], subject="1234", gitlab_realm="self-managed"
                ),
            ),
            {"detail": "Unauthorized to access homepage"},
            ["auth_duration_s", "token_issuer"],
        ),
        (
            # Invalid scope
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            403,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["unsupported_scope"],
                    subject="1234",
                    gitlab_realm="self-managed",
                ),
            ),
            {"detail": "Unauthorized to access homepage"},
            ["auth_duration_s", "token_issuer"],
        ),
        (
            # Unauthorized user
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            401,
            User(
                authenticated=False,
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="1234",
                    gitlab_realm="self-managed",
                ),
            ),
            {"error": "Forbidden by auth provider"},
            ["auth_duration_s", "auth_error_details", "token_issuer"],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "mismatch",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            401,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="1234",
                    gitlab_realm="self-managed",
                ),
            ),
            {"error": "Header mismatch 'X-Gitlab-Instance-Id'"},
            ["auth_duration_s", "auth_error_details", "token_issuer"],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "saas",
            },
            None,
            401,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="1234",
                    gitlab_realm="self-managed",
                ),
            ),
            {"error": "Header mismatch 'X-Gitlab-Realm'"},
            ["auth_duration_s", "auth_error_details", "token_issuer"],
        ),
        (
            # If JWT claim doesn't contain a value, the value header check is skipped
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "mismatch",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            200,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature1", "feature3"], gitlab_realm="self-managed"
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature3"],
            },
            ["auth_duration_s", "token_issuer"],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            401,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="1234",
                    issuer="gitlab-ai-gateway",
                    gitlab_realm="self-managed",
                ),
            ),
            {"error": "Header mismatch 'X-Gitlab-Global-User-Id'"},
            ["auth_duration_s", "auth_error_details", "token_issuer"],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            None,
            200,
            User(
                authenticated=True,
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="1234",
                    issuer="gitlab-ai-gateway",
                    gitlab_realm="self-managed",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature3"],
            },
            ["auth_duration_s", "token_issuer"],
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
        "scopes": [],
    }
