from contextlib import contextmanager
from typing import Optional

from prometheus_client import Gauge

from ai_gateway.tracking.errors import log_exception

METRIC_LABELS = ["model_engine", "model_name"]

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


class ModelRequestInstrumentator:
    class WatchContainer:
        def __init__(self, labels: dict[str, str], concurrency_limit: Optional[int]):
            self.labels = labels
            self.concurrency_limit = concurrency_limit

        def start(self):
            if self.concurrency_limit is not None:
                MAX_CONCURRENT_MODEL_INFERENCES.labels(**self.labels).set(
                    self.concurrency_limit
                )
            INFERENCE_IN_FLIGHT_GAUGE.labels(**self.labels).inc()

        def finish(self):
            INFERENCE_IN_FLIGHT_GAUGE.labels(**self.labels).dec()

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
            labels=self.labels, concurrency_limit=self.concurrency_limit
        )
        watcher.start()
        try:
            yield watcher
        except Exception as ex:
            log_exception(ex, self.labels)
            raise
        finally:
            if not stream:
                watcher.finish()
