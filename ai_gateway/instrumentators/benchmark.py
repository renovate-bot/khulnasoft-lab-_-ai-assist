import time
from contextlib import contextmanager
from enum import Enum

from prometheus_client import Histogram
from starlette_context import context

__all__ = ["benchmark", "KnownMetrics"]


class KnownMetrics(str, Enum):
    """Known list of Prometheus metrics."""

    POST_PROCESSING_DURATION = "post_processing_duration_s"


PROMETHEUS_METRICS: dict[KnownMetrics, Histogram] = {
    KnownMetrics.POST_PROCESSING_DURATION: Histogram(
        "code_suggestions_post_processing_duration_seconds",
        "Duration of post processing in seconds",
        ["model_engine", "model_name"],
    )
}


@contextmanager
def benchmark(metric_key: KnownMetrics, labels: dict[str, str]):
    """Benchmark and record elapsed time in log and Prometheus."""
    start_time = time.perf_counter()

    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Record in log
        context.data[metric_key.value] = elapsed_time

        # Record in Prometheus
        if prometheus_metric := PROMETHEUS_METRICS.get(metric_key):
            prometheus_metric.labels(**labels).observe(elapsed_time)
