from contextvars import ContextVar
from unittest.mock import MagicMock, Mock, patch

from anthropic import APIStatusError
from structlog.testing import capture_logs

from ai_gateway.models import AnthropicAPIStatusError
from ai_gateway.tracking.errors import log_exception


@patch("ai_gateway.tracking.errors.correlation_id")
@patch("ai_gateway.tracking.errors.traceback.format_exc")
def test_log_exception(mock_traceback, mock_correlation_id):
    mock_traceback.return_value = "dummy backtrace"
    mock_correlation_id.get.return_value = "123"

    with capture_logs() as cap_logs:
        log_exception(Exception("dummy message"))

    assert cap_logs[0]["event"].startswith("dummy message")
    assert cap_logs[0]["backtrace"] == "dummy backtrace"
    assert cap_logs[0]["correlation_id"] == "123"
    assert cap_logs[0]["status_code"] == None


@patch("ai_gateway.tracking.errors.correlation_id")
@patch("ai_gateway.tracking.errors.traceback.format_exc")
def test_log_exception_with_code(mock_traceback, mock_correlation_id):
    mock_traceback.return_value = "dummy backtrace with status code"
    mock_correlation_id.get.return_value = "123"

    with capture_logs() as cap_logs:
        ex = APIStatusError(
            message="dummy message with status code",
            response=MagicMock(),
            body=MagicMock(),
        )
        ex.status_code = 400
        ex = AnthropicAPIStatusError.from_exception(ex)
        log_exception(ex)

    assert cap_logs[0]["event"].startswith("400 dummy message with status code")
    assert cap_logs[0]["backtrace"] == "dummy backtrace with status code"
    assert cap_logs[0]["correlation_id"] == "123"
    assert cap_logs[0]["status_code"] == 400
