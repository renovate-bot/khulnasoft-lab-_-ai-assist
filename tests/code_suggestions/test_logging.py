from unittest import mock

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette_context.middleware import RawContextMiddleware
from structlog.testing import capture_logs

from ai_gateway.api.middleware import AccessLogMiddleware


def broken_page(request):
    raise RuntimeError("Something broke!")


app = Starlette(
    middleware=[
        Middleware(RawContextMiddleware),
        Middleware(AccessLogMiddleware, skip_endpoints=[]),
    ],
    routes=[Route("/", endpoint=broken_page, methods=["POST"])],
)
client = TestClient(app)


@mock.patch("ai_gateway.api.middleware.log_exception")
def test_x_gitlab_headers_logged_when_set(mock_log_exception):
    with capture_logs() as cap_logs, pytest.raises(RuntimeError):
        client.post(
            "/",
            headers={
                "X-Gitlab-Instance-Id": "ABC",
                "X-Gitlab-Global-User-Id": "DEF",
                "X-Gitlab-Host-Name": "awesome-org.com",
                "X-Gitlab-Saas-Namespace-Ids": "1,2,3",
                "X-Gitlab-Realm": "saas",
            },
            data={"foo": "bar"},
        )

        mock_log_exception.assert_called_once()

    assert cap_logs[0]["gitlab_instance_id"] == "ABC"
    assert cap_logs[0]["gitlab_global_user_id"] == "DEF"
    assert cap_logs[0]["gitlab_host_name"] == "awesome-org.com"
    assert cap_logs[0]["gitlab_saas_namespace_ids"] == "1,2,3"
    assert cap_logs[0]["gitlab_realm"] == "saas"


@mock.patch("ai_gateway.api.middleware.log_exception")
def test_x_gitlab_headers_not_logged_when_not_set(mock_log_exception):
    with capture_logs() as cap_logs, pytest.raises(RuntimeError):
        client.post("/", headers={}, data={"foo": "bar"})

        mock_log_exception.assert_called_once()

    assert cap_logs[0]["gitlab_instance_id"] is None
    assert cap_logs[0]["gitlab_global_user_id"] is None
    assert cap_logs[0]["gitlab_host_name"] is None
    assert cap_logs[0]["gitlab_realm"] is None


@mock.patch("ai_gateway.api.middleware.log_exception")
def test_exeption_capture(mock_log_exception):
    with capture_logs() as cap_logs, pytest.raises(RuntimeError):
        response = client.post("/", headers={}, data={"foo": "bar"})

        mock_log_exception.assert_called_once()

        assert response.status_code == 500

    assert cap_logs[0]["exception.message"] == "Something broke!"
    assert cap_logs[0]["exception.backtrace"].startswith("Traceback")
