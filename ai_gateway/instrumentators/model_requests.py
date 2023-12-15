from contextlib import contextmanager

from prometheus_client import Gauge

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

        def start(self):
            INFERENCE_IN_FLIGHT_GAUGE.labels(**self.labels).inc()

        def finish(self):
            INFERENCE_IN_FLIGHT_GAUGE.labels(**self.labels).dec()

    def __init__(self, model_engine: str, model_name: str):
        self.labels = {"model_engine": model_engine, "model_name": model_name}

    @contextmanager
    def watch(self, stream=False):
        watcher = ModelRequestInstrumentator.WatchContainer(self.labels)
        watcher.start()
        try:
            yield watcher
        finally:
            if not stream:
                watcher.finish()
