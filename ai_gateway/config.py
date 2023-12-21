import json
import os
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

__all__ = [
    "Config",
    "FastApiConfig",
    "AuthConfig",
    "PalmTextModelConfig",
    "Project",
    "FeatureFlags",
    "TrackingConfig",
]


class LoggingConfig(NamedTuple):
    level: str
    json: bool
    log_to_file: bool


class FastApiConfig(NamedTuple):
    docs_url: str
    openapi_url: str
    redoc_url: str
    api_host: str
    api_port: int
    metrics_host: str
    metrics_port: int
    uvicorn_logger: dict


class AuthConfig(NamedTuple):
    gitlab_base_url: str
    gitlab_api_base_url: str
    customer_portal_base_url: str
    bypass: bool


class ProfilingConfig(NamedTuple):
    enabled: bool
    verbose: int
    period_ms: int


class PalmTextModelConfig(NamedTuple):
    names: list[str]
    project: str
    location: str
    vertex_api_endpoint: str
    real_or_fake: str


class Project(NamedTuple):
    id: int
    full_name: str


class FeatureFlags(NamedTuple):
    is_third_party_ai_default: bool
    limited_access_third_party_ai: dict[int, Project]
    third_party_rollout_percentage: int
    code_suggestions_excl_post_proc: list


class TrackingConfig(NamedTuple):
    snowplow_enabled: bool
    snowplow_endpoint: str


class ModelConcurrencyConfig(NamedTuple):
    json_config: dict

    @property
    def parsed_config(self) -> dict:
        return json.loads(self.json_config)

    def for_model(self, engine: str, name: str) -> Union[int, None]:
        return self.parsed_config.get(engine, {}).get(name)


class Config:
    BOOLEAN_STATES = {
        "1": True,
        "yes": True,
        "true": True,
        "on": True,
        "0": False,
        "no": False,
        "false": False,
        "off": False,
    }

    STRUCTURED_LOGGING = {"version": 1, "disable_existing_loggers": False}

    @property
    def logging(self) -> LoggingConfig:
        return LoggingConfig(
            level=Config._get_value("LOG_LEVEL", "INFO"),
            json=Config._str_to_bool(Config._get_value("LOG_FORMAT_JSON", "True")),
            log_to_file=Config._get_value("LOG_TO_FILE", ""),
        )

    @property
    def fastapi(self) -> FastApiConfig:
        return FastApiConfig(
            docs_url=Config._get_value("FASTAPI_DOCS_URL", None),
            openapi_url=Config._get_value("FASTAPI_OPENAPI_URL", None),
            redoc_url=Config._get_value("FASTAPI_REDOC_URL", None),
            api_host=Config._get_value("FASTAPI_API_HOST", "0.0.0.0"),
            api_port=int(Config._get_value("FASTAPI_API_PORT", 5000)),
            metrics_host=Config._get_value("FASTAPI_API_METRICS_HOST", "0.0.0.0"),
            metrics_port=int(Config._get_value("FASTAPI_API_METRICS_PORT", 8082)),
            uvicorn_logger=Config.STRUCTURED_LOGGING,
        )

    @property
    def auth(self) -> AuthConfig:
        return AuthConfig(
            gitlab_base_url=Config._get_value("GITLAB_URL", "https://gitlab.com/"),
            gitlab_api_base_url=Config._get_value(
                "GITLAB_API_URL", "https://gitlab.com/api/v4/"
            ),
            customer_portal_base_url=Config._get_value(
                "CUSTOMER_PORTAL_BASE_URL", "https://customers.gitlab.com"
            ),
            bypass=Config._str_to_bool(
                Config._get_value("AUTH_BYPASS_EXTERNAL", "False")
            ),
        )

    @property
    def profiling(self) -> ProfilingConfig:
        return ProfilingConfig(
            enabled=Config._str_to_bool(
                Config._get_value("GOOGLE_CLOUD_PROFILER", "False")
            ),
            verbose=int(Config._get_value("GOOGLE_CLOUD_PROFILER_VERBOSE", 2)),
            period_ms=int(Config._get_value("GOOGLE_CLOUD_PROFILER_PERIOD_MS", 10)),
        )

    @property
    def feature_flags(self) -> FeatureFlags:
        limited_access = dict()
        if file_path := Config._get_value("F_THIRD_PARTY_AI_LIMITED_ACCESS", ""):
            projects = _read_projects_from_file(Path(file_path))
            limited_access = {project.id: project for project in projects}

        code_suggestions_excl_post_proc = []
        if feature_value := Config._get_value("F_CODE_SUGGESTIONS_EXCL_POST_PROC", ""):
            code_suggestions_excl_post_proc = feature_value.split(";")

        return FeatureFlags(
            is_third_party_ai_default=Config._str_to_bool(
                Config._get_value("F_IS_THIRD_PARTY_AI_DEFAULT", "False")
            ),
            limited_access_third_party_ai=limited_access,
            third_party_rollout_percentage=int(
                Config._get_value("F_THIRD_PARTY_ROLLOUT_PERCENTAGE", 0)
            ),
            code_suggestions_excl_post_proc=code_suggestions_excl_post_proc,
        )

    @property
    def palm_text_model(self) -> PalmTextModelConfig:
        names = []
        if s := Config._get_value("PALM_TEXT_MODEL_NAME", "text-bison,code-gecko"):
            names = s.split(",")

        return PalmTextModelConfig(
            names=names,
            project=Config._get_value("PALM_TEXT_PROJECT", "unreview-poc-390200e5"),
            location=Config._get_value("PALM_TEXT_LOCATION", "us-central1"),
            vertex_api_endpoint=Config._get_value(
                "VERTEX_API_ENDPOINT", "us-central1-aiplatform.googleapis.com"
            ),
            real_or_fake=Config._parse_fake_models(
                Config._get_value("USE_FAKE_MODELS", "False")
            ),
        )

    @property
    def tracking(self) -> TrackingConfig:
        return TrackingConfig(
            snowplow_enabled=Config._str_to_bool(
                Config._get_value("SNOWPLOW_ENABLED", "False")
            ),
            snowplow_endpoint=Config._get_value("SNOWPLOW_ENDPOINT", None),
        )

    @property
    def model_concurrency(self) -> ModelConcurrencyConfig:
        return ModelConcurrencyConfig(
            json_config=(Config._get_value("MODEL_ENGINE_CONCURRENCY_LIMITS", "{}"))
        )

    @staticmethod
    def _get_value(value: str, default: Optional[Any]):
        return os.environ.get(value, default)

    @staticmethod
    def _str_to_bool(value: str):
        if value.lower() not in Config.BOOLEAN_STATES:
            raise ValueError("Not a boolean: %s" % value)
        return Config.BOOLEAN_STATES[value.lower()]

    @staticmethod
    def _parse_fake_models(value: str) -> str:
        return "fake" if Config._str_to_bool(value) else "real"


def _read_projects_from_file(file_path: Path, sep: str = ",") -> list[Project]:
    projects = []
    with open(str(file_path), "r") as f:
        for line in f.readlines():
            line_split = line.strip().split(sep, maxsplit=2)
            projects.append(
                Project(
                    id=int(line_split[0]),
                    full_name=line_split[1],
                )
            )

    return projects
