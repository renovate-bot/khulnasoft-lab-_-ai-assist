import asyncio
import os
import sys
import traceback
from asyncio import AbstractEventLoop
from typing import Any

import structlog
from prometheus_client import Gauge

METRIC_LABELS = ["pid"]

__all__ = ["monitor_threads"]

log = structlog.stdlib.get_logger("threads")

AI_GATEWAY_THREADS_COUNT = Gauge(
    "ai_gateway_threads_count",
    "The number of active threads in the server",
    METRIC_LABELS,
)


async def monitor_threads(loop: AbstractEventLoop, interval: int):
    """
    Monitor the thread activities of the server.

    This task runs in the main event loop, meaning the web server will stop responding during the execution.
    Keep it light-weight, or consider running them in a separate thread in daemon mode.

    args:
        loop: The main event loop where the server is running.
        interval: Frequency of the thread activity scanning.
    """
    while loop.is_running():
        await asyncio.sleep(interval)

        pid = os.getpid()

        backtraces = []
        for thread_id, stack in sys._current_frames().items():
            thread_info: dict[str, Any] = {"thread_id": thread_id}
            lines: list[dict] = []

            for filename, lineno, name, line in traceback.extract_stack(stack):
                line_info = {"filename": filename, "lineno": lineno, "name": name}
                if line:
                    line_info["line"] = line.strip()
                lines.append(line_info)

            thread_info["lines"] = lines
            backtraces.append(thread_info)

        threads_count = len(backtraces)

        AI_GATEWAY_THREADS_COUNT.labels(pid=pid).set(threads_count)

        log.info(
            "Thread activities",
            pid=pid,
            threads_count=threads_count,
            stacktrace=backtraces,
        )
