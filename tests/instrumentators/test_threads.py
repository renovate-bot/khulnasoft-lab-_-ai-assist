import asyncio
from typing import Awaitable, Callable
from unittest import mock
from unittest.mock import ANY, Mock, patch

import pytest
from structlog.testing import capture_logs

from ai_gateway.instrumentators.threads import monitor_threads

max_loop_counter = 1


def mock_run_once():
    global max_loop_counter

    if max_loop_counter > 0:
        max_loop_counter -= 1
        return True
    else:
        return False


@pytest.mark.asyncio
@mock.patch("prometheus_client.Gauge.labels")
@mock.patch("ai_gateway.instrumentators.threads.asyncio.sleep")
async def test_monitor_threads(mock_sleep, mock_gauges):
    mock_loop = Mock()
    mock_loop.is_running = mock_run_once
    interval = 0.001

    with capture_logs() as cap_logs:
        await monitor_threads(mock_loop, interval=interval)

    mock_sleep.assert_any_await(interval)

    assert mock_gauges.mock_calls == [
        mock.call(pid=ANY),
        mock.call().set(ANY),
    ]

    assert cap_logs[0]["pid"] == ANY
    assert cap_logs[0]["threads_count"] == ANY
    assert cap_logs[0]["stacktrace"] == ANY
