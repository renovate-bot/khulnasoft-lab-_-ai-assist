from unittest import mock

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from structlog.testing import capture_logs

from codesuggestions.api.middleware import MiddlewareModelTelemetry


def homepage(request):
    return JSONResponse({"hello": "world"})


app = Starlette(
    middleware=[
        MiddlewareModelTelemetry(),
    ],
    routes=[Route("/", endpoint=homepage, methods=["POST"])],
)
client = TestClient(app)


@mock.patch("prometheus_client.Counter.inc")
def test_telemetry_capture_with_headers(mock_counter):
    headers = {
        "X-GitLab-CS-Accepts": "1",
        "X-GitLab-CS-Requests": "1",
        "X-GitLab-CS-Errors": "1",
    }

    with capture_logs() as cap_logs:
        response = client.post("/", headers=headers, data={"foo": "bar"})

    assert response.status_code == 200

    assert cap_logs == [
        {
            "accepted_request_count": 1,
            "total_request_count": 1,
            "error_request_count": 1,
            "event": "telemetry",
            "log_level": "info",
        }
    ]

    mock_counter.assert_has_calls(
        [
            mock.call(1),
            mock.call(1),
            mock.call(1),
        ]
    )


@mock.patch("prometheus_client.Counter.inc")
def test_telemetry_capture_without_headers(mock_counter):
    with capture_logs() as cap_logs:
        response = client.post("/", headers={}, data={"foo": "bar"})

    assert response.status_code == 200

    assert cap_logs == [
        {
            "accepted_request_count": 0,
            "total_request_count": 0,
            "error_request_count": 0,
            "event": "telemetry",
            "log_level": "info",
        }
    ]

    mock_counter.assert_has_calls(
        [
            mock.call(0),
            mock.call(0),
            mock.call(0),
        ]
    )
