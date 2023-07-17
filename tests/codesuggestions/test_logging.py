from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette_context.middleware import RawContextMiddleware
from structlog.testing import capture_logs

from codesuggestions.api.middleware import MiddlewareLogRequest


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


def test_exeption_capture():
    with capture_logs() as cap_logs:
        response = client.post("/", headers={}, data={"foo": "bar"})

    assert response.status_code == 500

    assert cap_logs[0]["exception"]["message"] == "Something broke!"
    assert cap_logs[0]["exception"]["backtrace"].startswith("Traceback")
