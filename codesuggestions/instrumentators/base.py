import structlog
import time

from contextlib import contextmanager
from starlette_context import context
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ValidationError

METRIC_LABELS = ["model_engine", "model_name"]

INFERENCE_COUNTER = Counter("code_suggestions_inference_requests",
                            "Number of times an inference request was made", METRIC_LABELS)

INFERENCE_HISTOGRAM = Histogram("code_suggestions_inference_request_duration_seconds",
                                "Duration of the inference request in seconds", METRIC_LABELS)

INFERENCE_PROMPT_HISTOGRAM = Histogram("code_suggestions_inference_prompt_size_bytes",
                                       "Size of the prompt of an inference request in bytes", METRIC_LABELS,
                                       buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096))

# TODO: Label accepts counter once the client starts sending model info
ACCEPTS_COUNTER = Counter("code_suggestions_accepts", "Accepts count by number")
REQUESTS_COUNTER = Counter("code_suggestions_requests", "Requests count by number", METRIC_LABELS)
ERRORS_COUNTER = Counter("code_suggestions_errors", "Errors count by number", METRIC_LABELS)

telemetry_logger = structlog.stdlib.get_logger("telemetry")


class TextGenModelInstrumentator:
    def __init__(self, model_engine: str, model_name: str):
        self.labels = {"model_engine": model_engine, "model_name": model_name}

    @contextmanager
    def watch(self, prompt: str):
        prompt_length = len(prompt)

        context["model_engine"] = self.labels["model_engine"]
        context["model_name"] = self.labels["model_name"]
        context["prompt_length"] = prompt_length

        INFERENCE_PROMPT_HISTOGRAM.labels(**self.labels).observe(prompt_length)
        INFERENCE_COUNTER.labels(**self.labels).inc()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            INFERENCE_HISTOGRAM.labels(**self.labels).observe(duration)

            context["inference_duration_s"] = duration


class ModelTelemetry(BaseModel):
    accepted_request_count: int = 0
    total_request_count: int = 0
    error_request_count: int = 0


class TelemetryInstrumentator:
    def __init__(self, accepted_request_count, total_request_count, error_request_count) -> None:
        self.accepted_request_count = accepted_request_count
        self.total_request_count = total_request_count
        self.error_request_count = error_request_count

    def __enter__(self):
        pass

    def __exit__(self, *exc):
        labels = {
            "model_engine": context.get("model_engine", ""),
            "model_name": context.get("model_name", ""),
        }

        try:
            fields = ModelTelemetry(
                accepted_request_count=self.accepted_request_count,
                total_request_count=self.total_request_count,
                error_request_count=self.error_request_count,
            )

            telemetry_logger.info("telemetry", **(fields.dict() | labels))

            ACCEPTS_COUNTER.inc(fields.accepted_request_count)
            REQUESTS_COUNTER.labels(**labels).inc(fields.total_request_count)
            ERRORS_COUNTER.labels(**labels).inc(fields.error_request_count)
        except ValidationError as e:
            telemetry_logger.error(f"failed to capture model telemetry: {e}", **labels)
