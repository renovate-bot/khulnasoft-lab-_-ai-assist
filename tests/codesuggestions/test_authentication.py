import json

import pytest
from starlette.applications import Starlette
from starlette.authentication import requires
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette_context.middleware import RawContextMiddleware
from structlog.testing import capture_logs

from codesuggestions.api.middleware import (
    MiddlewareAuthentication,
    MiddlewareLogRequest,
)
from codesuggestions.auth import User, UserClaims


@requires("authenticated")
def homepage(request):
    return JSONResponse()


class StubKeyAuthProvider:
    def authenticate(self, token):
        return User(
            authenticated=False,
            claims=UserClaims(
                is_third_party_ai_default=False,
            ),
        )


app = Starlette(
    middleware=[
        Middleware(RawContextMiddleware),
        MiddlewareLogRequest(),
        MiddlewareAuthentication(StubKeyAuthProvider(), None, False, None),
    ],
    routes=[Route("/", endpoint=homepage, methods=["POST"])],
)
client = TestClient(app)
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
]
forbidden_error = {"error": "Forbidden by auth provider"}


@pytest.mark.parametrize(
    ("headers", "data", "expected_response", "log_keys"),
    [
        (None, None, {"error": "No authorization header presented"}, []),
        (
            {"Authorization": "invalid"},
            None,
            {"error": "Invalid authorization header"},
            [],
        ),
        (
            {"Authorization": "Bearer 12345"},
            None,
            forbidden_error,
            ["auth_duration_s", "gitlab_realm"],
        ),
        # With project_id
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "1",
                "Content-Type": "application/json",
            },
            json.dumps({"project_id": 12345, "project_path": "a/b/c"}),
            forbidden_error,
            ["auth_duration_s", "gitlab_realm"],
        ),
        # Invalid JSON payload
        (
            {
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "1",
                "Content-Type": "application/json",
            },
            "this is not JSON{",
            forbidden_error,
            ["auth_duration_s", "gitlab_realm"],
        ),
    ],
)
def test_failed_authorization_logging(headers, data, expected_response, log_keys):
    with capture_logs() as cap_logs:
        response = client.post("/", headers=headers, data=data)

        assert response.status_code == 401
        assert response.json() == expected_response

        assert len(cap_logs) == 1
        assert cap_logs[0]["status_code"] == 401
        assert cap_logs[0]["method"] == "POST"
        assert set(cap_logs[0].keys()) == set(expected_log_keys + log_keys)
