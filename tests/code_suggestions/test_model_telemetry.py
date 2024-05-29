from unittest import mock
from unittest.mock import patch

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette_context import request_cycle_context
from structlog.testing import capture_logs

from ai_gateway.api.middleware import MiddlewareModelTelemetry


def homepage(request):
    return JSONResponse({"hello": "world"})


app = Starlette(
    middleware=[
        MiddlewareModelTelemetry(),
    ],
    routes=[Route("/", endpoint=homepage, methods=["POST"])],
)
client = TestClient(app)
context = {"model_engine": "codegen", "model_name": "ensemble"}


@mock.patch("prometheus_client.Counter.inc")
def test_telemetry_capture_with_headers(mock_counter):
    headers = {
        "X-GitLab-CS-Accepts": "1",
        "X-GitLab-CS-Requests": "1",
        "X-GitLab-CS-Errors": "1",
    }

    with capture_logs() as cap_logs, request_cycle_context(context):
        response = client.post("/", headers=headers, data={"foo": "bar"})

    assert response.status_code == 200

    assert cap_logs == [
        {
            "accepts": 1,
            "requests": 1,
            "errors": 1,
            "event": "telemetry",
            "log_level": "info",
            "model_engine": "codegen",
            "model_name": "ensemble",
            "lang": None,
            "experiments": None,
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
    with capture_logs() as cap_logs, request_cycle_context(context):
        response = client.post("/", headers={}, data={"foo": "bar"})

    assert response.status_code == 200

    assert len(cap_logs) == 0
    assert mock_counter.call_count == 0


@mock.patch("prometheus_client.Counter.inc")
def test_telemetry_capture_invalid_headers(mock_counter):
    headers = {
        "X-GitLab-CS-Accepts": "one",
        "X-GitLab-CS-Requests": "more",
        "X-GitLab-CS-Errors": "time",
    }

    with patch("ai_gateway.api.middleware.log_exception") as mock_log_exception:
        response = client.post("/", headers=headers, data={"foo": "bar"})

        args, _ = mock_log_exception.call_args
        assert isinstance(args[0], ValidationError)

    assert response.status_code == 200
    assert mock_counter.call_count == 0
