import time
from contextlib import contextmanager
from typing import Any, Optional

import structlog
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, constr
from starlette_context import context

METRIC_LABELS = ["model_engine", "model_name"]
TELEMETRY_LABELS = METRIC_LABELS + ["lang"]

INFERENCE_COUNTER = Counter(
    "code_suggestions_inference_requests",
    "Number of times an inference request was made",
    METRIC_LABELS,
)

INFERENCE_HISTOGRAM = Histogram(
    "code_suggestions_inference_request_duration_seconds",
    "Duration of the inference request in seconds",
    METRIC_LABELS,
)

INFERENCE_PROMPT_HISTOGRAM = Histogram(
    "code_suggestions_inference_prompt_size_bytes",
    "Size of the prompt of an inference request in bytes",
    METRIC_LABELS,
    buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096),
)

ACCEPTS_COUNTER = Counter(
    "code_suggestions_accepts", "Accepts count by number", TELEMETRY_LABELS
)
REQUESTS_COUNTER = Counter(
    "code_suggestions_requests", "Requests count by number", TELEMETRY_LABELS
)
ERRORS_COUNTER = Counter(
    "code_suggestions_errors", "Errors count by number", TELEMETRY_LABELS
)

telemetry_logger = structlog.stdlib.get_logger("telemetry")


class TextGenModelInstrumentator:
    class WatchContainer:
        def __init__(self, **kwargs: Any):
            self.__dict__.update(**kwargs)

        def register_model_exception(self, message: str, status_code: int):
            self.__dict__.update(
                {
                    "model_exception_message": message,
                    "model_exception_status_code": status_code,
                }
            )

        def register_prompt_symbols(self, symbol_map: dict[str, int]):
            self.__dict__.update({"prompt_symbols": symbol_map})

        def dict(self) -> dict:
            return self.__dict__

    def __init__(self, model_engine: str, model_name: str):
        self.labels = {"model_engine": model_engine, "model_name": model_name}

    @contextmanager
    def watch(self, prompt: str, **kwargs: Any):
        prompt_length = len(prompt)

        context["model_engine"] = self.labels["model_engine"]
        context["model_name"] = self.labels["model_name"]
        context["prompt_length"] = prompt_length

        INFERENCE_PROMPT_HISTOGRAM.labels(**self.labels).observe(prompt_length)
        INFERENCE_COUNTER.labels(**self.labels).inc()

        watch_container = TextGenModelInstrumentator.WatchContainer(**kwargs)
        start_time = time.perf_counter()

        try:
            yield watch_container
        finally:
            duration = time.perf_counter() - start_time
            INFERENCE_HISTOGRAM.labels(**self.labels).observe(duration)

            context["inference_duration_s"] = duration
            context.update(watch_container.dict())


class Telemetry(BaseModel):
    # TODO: Once the header telemetry format is removed, we can unmark these as optional
    model_engine: Optional[constr(max_length=50)]
    model_name: Optional[constr(max_length=50)]
    lang: Optional[constr(max_length=50)]
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
                    "model_engine": stats.model_engine
                    or context.get("model_engine", ""),
                    "model_name": stats.model_name or context.get("model_name", ""),
                    "lang": stats.lang,
                }

                telemetry_logger.info("telemetry", **(stats.dict() | labels))

                ACCEPTS_COUNTER.labels(**labels).inc(stats.accepts)
                REQUESTS_COUNTER.labels(**labels).inc(stats.requests)
                ERRORS_COUNTER.labels(**labels).inc(stats.errors)
