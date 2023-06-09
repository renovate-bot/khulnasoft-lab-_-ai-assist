import structlog
import time

from contextlib import contextmanager

from prometheus_client import Counter, Histogram

access_logger = structlog.stdlib.get_logger("api.access")

METRIC_LABELS = ["model_engine", "model_name"]

INFERENCE_COUNTER = Counter("code_suggestions_inference_requests",
                            "Number of times an inference request was made", METRIC_LABELS)

INFERENCE_HISTOGRAM = Histogram("code_suggestions_inference_request_duration_seconds",
                                "Duration of the inference request in seconds", METRIC_LABELS)

INFERENCE_PROMPT_HISTOGRAM = Histogram("code_suggestions_inference_prompt_size_bytes",
                                       "Size of the prompt of an inference request in bytes", METRIC_LABELS,
                                       buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096))


class TextGenModelInstrumentator:
    def __init__(self, model_engine: str, model_name: str):
        self.labels = {"model_engine": model_engine, "model_name": model_name}

    @contextmanager
    def watch(self, prompt: str):
        prompt_length = len(prompt)
        log_labels = self.labels | {"prompt_length": prompt_length}

        access_logger.info("Starting inference request", **log_labels)

        INFERENCE_PROMPT_HISTOGRAM.labels(**self.labels).observe(prompt_length)
        INFERENCE_COUNTER.labels(**self.labels).inc()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            INFERENCE_HISTOGRAM.labels(**self.labels).observe(duration)

            log_labels = log_labels | {"inference_duration_s": duration}
            access_logger.info("Finished inference request", **log_labels)
