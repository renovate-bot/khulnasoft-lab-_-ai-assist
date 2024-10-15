from typing import Annotated

import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse
from structlog.testing import capture_logs

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.middleware import MiddlewareAuthentication
from ai_gateway.cloud_connector import User, UserClaims

router = APIRouter(
    prefix="",
    tags=["something"],
)


@router.post("/")
def homepage(
    request: Request, current_user: Annotated[StarletteUser, Depends(get_current_user)]
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
    "response_start_duration_s",
    "first_chunk_duration_s",
    "request_arrived_at",
    "content_type",
    "user_agent",
    "event",
    "log_level",
    "gitlab_realm",
    "gitlab_instance_id",
    "gitlab_global_user_id",
    "gitlab_host_name",
    "gitlab_version",
    "gitlab_saas_duo_pro_namespace_ids",
    "gitlab_language_server_version",
    "gitlab_duo_seat_count",
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
            ["auth_duration_s", "auth_error_details", "http_exception_details"],
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
            ["auth_duration_s", "auth_error_details", "http_exception_details"],
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
            [
                "auth_duration_s",
                "auth_error_details",
                "http_exception_details",
                "token_issuer",
            ],
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
                    gitlab_instance_id="1234",
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
                    gitlab_instance_id="1234",
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
                    gitlab_instance_id="1234",
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
                    scopes=["feature1"],
                    subject="1234",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="1234",
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
                    scopes=["feature3"],
                    subject="1234",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="1234",
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
                    gitlab_instance_id="1234",
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
                    gitlab_instance_id="1234",
                ),
            ),
            {"error": "Forbidden by auth provider"},
            [
                "auth_duration_s",
                "auth_error_details",
                "token_issuer",
                "http_exception_details",
            ],
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
                    gitlab_instance_id="1234",
                ),
            ),
            {"error": "Header mismatch 'X-Gitlab-Instance-Id'"},
            [
                "auth_duration_s",
                "auth_error_details",
                "token_issuer",
                "http_exception_details",
            ],
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
                    gitlab_instance_id="1234",
                ),
            ),
            {"error": "Header mismatch 'X-Gitlab-Realm'"},
            [
                "auth_duration_s",
                "auth_error_details",
                "token_issuer",
                "http_exception_details",
            ],
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
                    scopes=["feature1", "feature3"],
                    gitlab_realm="self-managed",
                    gitlab_instance_id="",
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
                    gitlab_instance_id="1234",
                ),
            ),
            {"error": "Header mismatch 'X-Gitlab-Global-User-Id'"},
            [
                "auth_duration_s",
                "auth_error_details",
                "token_issuer",
                "http_exception_details",
            ],
        ),
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-Gitlab-Instance-Id": "1234",
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
                    gitlab_instance_id="1234",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature3"],
            },
            ["auth_duration_s", "token_issuer"],
        ),
        # 'duo_seat_count' claim is non-empty, 'X-Gitlab-Duo-Seat-Count' is present and matches the claim
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-Gitlab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Duo-Seat-Count": "333",
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
                    duo_seat_count="333",
                    gitlab_instance_id="1234",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature3"],
            },
            ["auth_duration_s", "token_issuer"],
        ),
        # 'gitlab_instance_id' claim is non-empty, 'X-Gitlab-Instance-Id' is missing
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-Gitlab-Duo-Seat-Count": "333",
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
                    duo_seat_count="333",
                    gitlab_instance_id="1234",
                ),
            ),
            {"error": "Header mismatch 'X-Gitlab-Instance-Id'"},
            [
                "auth_duration_s",
                "auth_error_details",
                "token_issuer",
                "http_exception_details",
            ],
        ),
        # 'gitlab_instance_id' claim is non-empty, 'X-Gitlab-Instance-Id' is present but does not match the claim
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-Gitlab-Instance-Id": "mismatch",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Duo-Seat-Count": "333",
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
                    duo_seat_count="333",
                    gitlab_instance_id="1234",
                ),
            ),
            {"error": "Header mismatch 'X-Gitlab-Instance-Id'"},
            [
                "auth_duration_s",
                "auth_error_details",
                "token_issuer",
                "http_exception_details",
            ],
        ),
        # 'gitlab_instance_id' claim is non-empty, 'X-Gitlab-Instance-Id' is present and matches the claim
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-Gitlab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Duo-Seat-Count": "333",
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
                    duo_seat_count="333",
                    gitlab_instance_id="1234",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature3"],
            },
            [
                "auth_duration_s",
                "token_issuer",
            ],
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


@pytest.mark.parametrize(
    (
        "headers",
        "data",
        "expected_status_code",
        "auth_user",
        "expected_response",
        "expected_error_message",
        "log_keys",
    ),
    [
        # 'duo_seat_count' claim is non-empty, 'X-Gitlab-Duo-Seat-Count' is missing
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-Gitlab-Instance-Id": "1234",
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
                    duo_seat_count="333",
                    gitlab_instance_id="1234",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature3"],
            },
            "Header is missing: 'X-Gitlab-Duo-Seat-Count'",
            [
                "auth_duration_s",
                "token_issuer",
            ],
        ),
        # 'duo_seat_count' claim is non-empty, 'X-Gitlab-Duo-Seat-Count' is present but does not match the claim
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-Gitlab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Duo-Seat-Count": "777",
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
                    duo_seat_count="333",
                    gitlab_instance_id="1234",
                ),
            ),
            {
                "authenticated": True,
                "is_debug": False,
                "scopes": ["feature1", "feature3"],
            },
            "Header mismatch 'X-Gitlab-Duo-Seat-Count'",
            [
                "auth_duration_s",
                "token_issuer",
            ],
        ),
    ],
)
def test_failed_duo_seat_count_validation_logging(
    # We have a separate spec for this, since this is a unique case, where validation fails,
    # we log an error, but we still want the request to pass. Extracted from `test_failed_authorization_logging`
    # See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/672
    mock_client,
    headers,
    data,
    expected_status_code,
    expected_error_message,
    expected_response,
    log_keys,
):
    with capture_logs() as cap_logs:
        response = mock_client.post("/", headers=headers, data=data)

        assert response.status_code == expected_status_code
        assert response.json() == expected_response

        assert len(cap_logs) == 2  # Regular logging and exception logging
        assert cap_logs[0]["event"] == expected_error_message
        assert cap_logs[1]["status_code"] == expected_status_code
        assert cap_logs[1]["method"] == "POST"
        assert set(cap_logs[1].keys()) == set(expected_log_keys + log_keys)


def test_bypass_auth(fast_api_router, stub_auth_provider):
    middlewares = [
        MiddlewareAuthentication(stub_auth_provider, True, False),
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


def test_bypass_external_with_header(fast_api_router, stub_auth_provider):
    middlewares = [
        MiddlewareAuthentication(stub_auth_provider, False, True),
    ]
    app = FastAPI(middleware=middlewares)
    app.include_router(fast_api_router)
    mock_client = TestClient(app)

    response = mock_client.post("/", headers={"Bypass-Auth": "true"})

    assert response.json() == {
        "authenticated": True,
        "is_debug": True,
        "scopes": [],
    }
