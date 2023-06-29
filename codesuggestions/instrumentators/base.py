import structlog
import time

from contextlib import contextmanager
from starlette_context import context
from typing import Optional, Any
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, constr

METRIC_LABELS = ["model_engine", "model_name"]

INFERENCE_COUNTER = Counter("code_suggestions_inference_requests",
                            "Number of times an inference request was made", METRIC_LABELS)

INFERENCE_HISTOGRAM = Histogram("code_suggestions_inference_request_duration_seconds",
                                "Duration of the inference request in seconds", METRIC_LABELS)

INFERENCE_PROMPT_HISTOGRAM = Histogram("code_suggestions_inference_prompt_size_bytes",
                                       "Size of the prompt of an inference request in bytes", METRIC_LABELS,
                                       buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096))

ACCEPTS_COUNTER = Counter("code_suggestions_accepts", "Accepts count by number", METRIC_LABELS)
REQUESTS_COUNTER = Counter("code_suggestions_requests", "Requests count by number", METRIC_LABELS)
ERRORS_COUNTER = Counter("code_suggestions_errors", "Errors count by number", METRIC_LABELS)

telemetry_logger = structlog.stdlib.get_logger("telemetry")


class TextGenModelInstrumentator:
    def __init__(self, model_engine: str, model_name: str):
        self.labels = {"model_engine": model_engine, "model_name": model_name}

    @contextmanager
    def watch(self, prompt: str, **kwargs: Any):
        prompt_length = len(prompt)

        context["model_engine"] = self.labels["model_engine"]
        context["model_name"] = self.labels["model_name"]
        context["prompt_length"] = prompt_length
        context.update(**kwargs)

        INFERENCE_PROMPT_HISTOGRAM.labels(**self.labels).observe(prompt_length)
        INFERENCE_COUNTER.labels(**self.labels).inc()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            INFERENCE_HISTOGRAM.labels(**self.labels).observe(duration)

            context["inference_duration_s"] = duration


class Telemetry(BaseModel):
    # TODO: Once the header telemetry format is removed, we can unmark these as optional
    model_engine: Optional[constr(max_length=50)]
    model_name: Optional[constr(max_length=50)]
    requests: int
    accepts: int
    errors: int


class TelemetryInstrumentator:
    @contextmanager
    def watch(self, telemetry: list[Telemetry]):
        try:
            yield
        finally:
            for stats in telemetry:
                # TODO: Once header telemetry is deprecated, we can remove the `or`
                labels = {
                    "model_engine": stats.model_engine or context.get("model_engine", ""),
                    "model_name": stats.model_name or context.get("model_name", ""),
                }

                telemetry_logger.info("telemetry", **(stats.dict() | labels))

                ACCEPTS_COUNTER.labels(**labels).inc(stats.accepts)
                REQUESTS_COUNTER.labels(**labels).inc(stats.requests)
                ERRORS_COUNTER.labels(**labels).inc(stats.errors)
