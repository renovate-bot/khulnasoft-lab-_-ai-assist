import re
import time
from contextlib import contextmanager
from typing import Annotated, Any, Optional

import structlog
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette_context import context

from ai_gateway.experimentation import ExperimentTelemetry
from ai_gateway.models import SafetyAttributes

EXPERIMENT_LABELS = ["exp_names", "exp_variants"]
METRIC_LABELS = ["model_engine", "model_name"]
TELEMETRY_LABELS = METRIC_LABELS + ["lang"] + EXPERIMENT_LABELS
PROMPT_LABELS = METRIC_LABELS + ["component"]

INFERENCE_COUNTER = Counter(
    "code_suggestions_inference_requests",
    "Number of times an inference request was made",
    METRIC_LABELS,
)

INFERENCE_HISTOGRAM = Histogram(
    "code_suggestions_inference_request_duration_seconds",
    "Duration of the inference request in seconds",
    METRIC_LABELS,
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60),
)

INFERENCE_PROMPT_HISTOGRAM = Histogram(
    "code_suggestions_inference_prompt_size_bytes",
    "Size of the prompt of an inference request in bytes",
    PROMPT_LABELS,
    buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096),
)

INFERENCE_PROMPT_TOKENS_HISTOGRAM = Histogram(
    "code_suggestions_inference_prompt_size_tokens",
    "Size of the prompt of an inference request in tokens",
    PROMPT_LABELS,
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

# Cost tracking metric for Vertex requests.
#
# NOTE: This counter is currently incremented both in gitlab-rails (AI abstraction layer) and here
# since different vendors support different use cases. Keep this definition consistent with
# https://gitlab.com/gitlab-org/gitlab/-/blob/837b7c68aaecf4b808d493a8bf08aab00ccb20f0/ee/lib/gitlab/llm/open_ai/client.rb#L150
CLOUD_COST_COUNTER_LABELS = ["item", "unit", "vendor", "model", "feature_category"]
CLOUD_COST_COUNTER = Counter(
    "gitlab_cloud_cost_spend_entry_total",
    "Number of units spent per vendor entry",
    CLOUD_COST_COUNTER_LABELS,
)

telemetry_logger = structlog.stdlib.get_logger("telemetry")

WHITESPACE_REGEX = re.compile(r"\s+")


def remove_whitespace(text: str) -> str:
    return WHITESPACE_REGEX.sub("", text)


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

        # Track model output length both in terms of unaltered character count as well as
        # with whitespace stripped out. The latter is used to calculate cloud provider cost.
        def register_model_output_length(self, model_output: str):
            self.__dict__.update(
                {
                    "model_output_length": len(model_output),
                    "model_output_length_stripped": len(
                        remove_whitespace(model_output)
                    ),
                }
            )

        def register_experiments(self, experiments: list[ExperimentTelemetry]):
            included_experiments = []
            for exp in experiments:
                entry = {
                    "name": exp.name,
                    "variant": exp.variant,
                }
                included_experiments.append(entry)
            self.__dict__.update({"experiments": included_experiments})

        def register_model_score(self, model_score: float):
            self.__dict__.update({"model_output_score": model_score})

        def register_lang(self, lang_id, editor_lang: Optional[str]):
            lang = lang_id.name.lower() if lang_id else ""

            self.__dict__.update({"lang": lang, "editor_lang": editor_lang})

        def register_is_discarded(self):
            self.__dict__.update({"discarded": True})

        def register_safety_attributes(self, safety_attributes: SafetyAttributes):
            self.__dict__.update({"blocked": safety_attributes.blocked})

            if safety_attributes.errors:
                self.__dict__.update({"error_codes": safety_attributes.errors})

            if safety_attributes.categories:
                self.__dict__.update(
                    {"safety_categories": safety_attributes.categories}
                )

        def dict(self) -> dict:
            return self.__dict__

    def __init__(self, model_engine: str, model_name: str):
        self.labels = {"model_engine": model_engine, "model_name": model_name}

    @contextmanager
    def watch(self, prompt, **kwargs: Any):
        prompt_string = f"{prompt.prefix}{prompt.suffix if prompt.suffix else ''}"
        prompt_length = len(prompt_string)
        prompt_length_stripped = len(remove_whitespace(prompt_string))

        context["model_engine"] = self.labels["model_engine"]
        context["model_name"] = self.labels["model_name"]
        context["prompt_length"] = prompt_length
        context["prompt_length_stripped"] = prompt_length_stripped

        for name, md in prompt.metadata.components.items():
            labels = self.labels.copy()
            labels["component"] = name
            INFERENCE_PROMPT_HISTOGRAM.labels(**labels).observe(md.length)
            INFERENCE_PROMPT_TOKENS_HISTOGRAM.labels(**labels).observe(md.length_tokens)

        INFERENCE_COUNTER.labels(**self.labels).inc()
        self._track_model_cost("input", prompt_length_stripped)

        watch_container = TextGenModelInstrumentator.WatchContainer(**kwargs)
        start_time = time.perf_counter()

        try:
            yield watch_container
        finally:
            duration = time.perf_counter() - start_time
            INFERENCE_HISTOGRAM.labels(**self.labels).observe(duration)

            container_dict = watch_container.dict()
            self._track_model_cost(
                "output", container_dict.get("model_output_length_stripped", 0)
            )

            context["inference_duration_s"] = duration
            context.update(container_dict)

    def _track_model_cost(self, kind, character_count):
        labels = {
            "item": f"completions/completion/{kind}",
            "unit": "characters",
            "vendor": self.labels["model_engine"],
            "model": self.labels["model_name"],
            "feature_category": "code_suggestions",
        }
        CLOUD_COST_COUNTER.labels(**labels).inc(character_count)


class Telemetry(BaseModel):
    # Opt out protected namespace "model_" (https://github.com/pydantic/pydantic/issues/6322).
    model_config = ConfigDict(protected_namespaces=())

    # TODO: Once the header telemetry format is removed, we can unmark these as optional
    model_engine: Optional[Annotated[str, StringConstraints(max_length=50)]] = None
    model_name: Optional[Annotated[str, StringConstraints(max_length=50)]] = None
    lang: Optional[Annotated[str, StringConstraints(max_length=50)]] = None
    experiments: Optional[list[ExperimentTelemetry]] = None
    requests: int
    accepts: int
    errors: int


def _format_experiment_telemetry(
    experiments: Optional[list[ExperimentTelemetry]],
) -> dict:
    if not experiments:
        return {"exp_names": None, "exp_variants": None}
    return {
        "exp_names": ",".join(exp.name for exp in experiments),
        "exp_variants": ",".join(str(exp.variant) for exp in experiments),
    }


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

                # add stringified exp data after the telemetry_logger call,
                # since this data belongs to the Prometheus counters only
                labels.update(_format_experiment_telemetry(stats.experiments))

                ACCEPTS_COUNTER.labels(**labels).inc(stats.accepts)
                REQUESTS_COUNTER.labels(**labels).inc(stats.requests)
                ERRORS_COUNTER.labels(**labels).inc(stats.errors)
