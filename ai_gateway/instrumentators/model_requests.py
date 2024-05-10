import time
from contextlib import contextmanager
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

from ai_gateway.api.feature_category import get_feature_category
from ai_gateway.tracking.errors import log_exception

METRIC_LABELS = ["model_engine", "model_name"]
INFERENCE_DETAILS = METRIC_LABELS + ["error", "streaming", "feature_category"]

INFERENCE_IN_FLIGHT_GAUGE = Gauge(
    "model_inferences_in_flight",
    "The number of in flight inferences running",
    METRIC_LABELS,
)

MAX_CONCURRENT_MODEL_INFERENCES = Gauge(
    "model_inferences_max_concurrent",
    "The maximum number of inferences we can run concurrently on a model",
    METRIC_LABELS,
)

# The counter and histogram from `instrumentators/base.py` can be removed once
# the SLIs stop using these. Then all requests, not just the code-suggestion ones
# will be instrumented
# We'll remove this as part of https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/441
INFERENCE_COUNTER = Counter(
    "model_inferences_total",
    "The total number of inferences on a model with a label",
    INFERENCE_DETAILS,
)

INFERENCE_DURATION_S = Histogram(
    "inference_request_duration_seconds",
    "Duration of the inference request in seconds",
    INFERENCE_DETAILS,
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60),
)


class ModelRequestInstrumentator:
    class WatchContainer:
        def __init__(
            self,
            labels: dict[str, str],
            concurrency_limit: Optional[int],
            streaming: bool,
        ):
            self.labels = labels
            self.concurrency_limit = concurrency_limit
            self.error = False
            self.streaming = streaming
            self.start_time = None

        def start(self):
            self.start_time = time.perf_counter()

            if self.concurrency_limit is not None:
                MAX_CONCURRENT_MODEL_INFERENCES.labels(**self.labels).set(
                    self.concurrency_limit
                )
            INFERENCE_IN_FLIGHT_GAUGE.labels(**self.labels).inc()

        def register_error(self):
            self.error = True

        def finish(self):
            INFERENCE_IN_FLIGHT_GAUGE.labels(**self.labels).dec()

            duration = time.perf_counter() - self.start_time
            detail_labels = self._detail_labels()

            INFERENCE_COUNTER.labels(**detail_labels).inc()
            INFERENCE_DURATION_S.labels(**detail_labels).observe(duration)

        async def afinish(self):
            self.finish()

        def _detail_labels(self) -> dict[str, str]:

            detail_labels = {
                "error": "yes" if self.error else "no",
                "streaming": "yes" if self.streaming else "no",
                "feature_category": get_feature_category(),
            }
            return {**self.labels, **detail_labels}

    def __init__(
        self,
        model_engine: str,
        model_name: str,
        concurrency_limit: Optional[int],
    ):
        self.labels = {"model_engine": model_engine, "model_name": model_name}
        self.concurrency_limit = concurrency_limit

    @contextmanager
    def watch(self, stream=False):
        watcher = ModelRequestInstrumentator.WatchContainer(
            labels=self.labels,
            concurrency_limit=self.concurrency_limit,
            streaming=stream,
        )
        watcher.start()
        try:
            yield watcher
        except Exception as ex:
            log_exception(ex, self.labels)
            watcher.register_error()
            raise
        finally:
            if not stream:
                watcher.finish()
