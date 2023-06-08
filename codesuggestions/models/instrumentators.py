import structlog
import time

from prometheus_client import Counter, Histogram

access_logger = structlog.stdlib.get_logger("api.access")

INFERENCE_COUNTER = Counter("code_suggestions_inference_requests",
                            "Number of times an inference request was made", ["model_engine", "model_name"])

INFERENCE_HISTOGRAM = Histogram("code_suggestions_inference_request_duration_seconds",
                                "Duration of the inference request in seconds", ["model_engine", "model_name"])


class TextGenModelInstrumentator:
    def __init__(self, model_engine: str, model_name: str):
        self.labels = {"model_engine": model_engine, "model_name": model_name}

    def __enter__(self):
        access_logger.info("Starting inference request", **self.labels)

        INFERENCE_COUNTER.labels(**self.labels).inc()
        self.start_time = time.perf_counter()

    def __exit__(self, *exc):
        duration = time.perf_counter() - self.start_time
        INFERENCE_HISTOGRAM.labels(**self.labels).observe(duration)

        log_labels = self.labels | {"inference_duration_s": duration}
        access_logger.info("Finished inference request", **log_labels)
