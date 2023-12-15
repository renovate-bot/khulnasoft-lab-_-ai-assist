from contextlib import contextmanager
from prometheus_client import Gauge
from typing import AsyncIterator

METRIC_LABELS = ["model_engine", "model_name"]

INFERENCE_IN_FLIGHT_GAUGE = Gauge(
    "model_inferences_in_flight",
    "The number of in flight inferences running",
    METRIC_LABELS,
)


class ModelRequestInstrumentator:
    class WatchContainer:
        def __init__(self, labels: dict[str, str]):
            self.labels = labels

        def _start(self):
            INFERENCE_IN_FLIGHT_GAUGE.labels(**self.labels).inc()

        def finish(self):
            INFERENCE_IN_FLIGHT_GAUGE.labels(**self.labels).dec()

        async def handle_and_finish_async(self, iterator: AsyncIterator):
            try:
                async for item in iterator:
                    yield item
            finally:
                self.finish()

    def __init__(self, model_engine: str, model_name: str):
        self.labels = {"model_engine": model_engine, "model_name": model_name}

    @contextmanager
    def watch(self, asyncOperation=False):
        watcher = ModelRequestInstrumentator.WatchContainer(self.labels)
        watcher._start()
        try:
            yield watcher
        finally:
            if not asyncOperation:
                watcher.finish()
