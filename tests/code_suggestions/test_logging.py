from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette_context.middleware import RawContextMiddleware
from structlog.testing import capture_logs

from ai_gateway.api.middleware import MiddlewareLogRequest


def broken_page(request):
    raise RuntimeError("Something broke!")


app = Starlette(
    middleware=[
        Middleware(RawContextMiddleware),
        MiddlewareLogRequest(),
    ],
    routes=[Route("/", endpoint=broken_page, methods=["POST"])],
)
client = TestClient(app)


def test_x_gitlab_headers_logged_when_set():
    with capture_logs() as cap_logs:
        client.post(
            "/",
            headers={
                "X-Gitlab-Instance-Id": "ABC",
                "X-Gitlab-Global-User-Id": "DEF",
                "X-Gitlab-Host-Name": "awesome-org.com",
                "X-Gitlab-Realm": "saas",
            },
            data={"foo": "bar"},
        )

    assert cap_logs[0]["gitlab_instance_id"] == "ABC"
    assert cap_logs[0]["gitlab_global_user_id"] == "DEF"
    assert cap_logs[0]["gitlab_host_name"] == "awesome-org.com"
    assert cap_logs[0]["gitlab_realm"] == "saas"


def test_x_gitlab_headers_not_logged_when_not_set():
    with capture_logs() as cap_logs:
        client.post("/", headers={}, data={"foo": "bar"})

    assert cap_logs[0]["gitlab_instance_id"] is None
    assert cap_logs[0]["gitlab_global_user_id"] is None
    assert cap_logs[0]["gitlab_host_name"] is None
    assert cap_logs[0]["gitlab_realm"] is None


def test_exeption_capture():
    with capture_logs() as cap_logs:
        response = client.post("/", headers={}, data={"foo": "bar"})

    assert response.status_code == 500

    assert cap_logs[0]["exception"]["message"] == "Something broke!"
    assert cap_logs[0]["exception"]["backtrace"].startswith("Traceback")
