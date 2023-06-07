import os

from typing import Optional, Any, NamedTuple

__all__ = [
    "Config",
    "TritonConfig",
    "FastApiConfig",
    "AuthConfig",
]


class TritonConfig(NamedTuple):
    host: str
    port: int


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
    bypass: bool


class ProfilingConfig(NamedTuple):
    enabled: bool
    verbose: int
    period_ms: int


class Config:
    BOOLEAN_STATES = {
        '1': True, 'yes': True, 'true': True, 'on': True,
        '0': False, 'no': False, 'false': False, 'off': False
    }

    STRUCTURED_LOGGING = {
        "version": 1,
        "disable_existing_loggers": False
    }

    @property
    def triton(self) -> TritonConfig:
        return TritonConfig(
            host=Config._get_value("TRITON_HOST", "triton"),
            port=int(Config._get_value("TRITON_PORT", 8001)),
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
            uvicorn_logger=Config.STRUCTURED_LOGGING
        )

    @property
    def auth(self) -> AuthConfig:
        return AuthConfig(
            gitlab_base_url=Config._get_value("GITLAB_URL", "https://gitlab.com/"),
            gitlab_api_base_url=Config._get_value("GITLAB_API_URL", "https://gitlab.com/api/v4/"),
            bypass=Config._str_to_bool(Config._get_value("AUTH_BYPASS_EXTERNAL", "False"))
        )

    @property
    def profiling(self) -> ProfilingConfig:
        return ProfilingConfig(
            enabled=Config._str_to_bool(Config._get_value("GOOGLE_CLOUD_PROFILER", "False")),
            verbose=int(Config._get_value("GOOGLE_CLOUD_PROFILER_VERBOSE", 2)),
            period_ms=int(Config._get_value("GOOGLE_CLOUD_PROFILER_PERIOD_MS", 10)),
        )

    @staticmethod
    def _get_value(value: str, default: Optional[Any]):
        return os.environ.get(value, default)

    @staticmethod
    def _str_to_bool(value: str):
        if value.lower() not in Config.BOOLEAN_STATES:
            raise ValueError('Not a boolean: %s' % value)
        return Config.BOOLEAN_STATES[value.lower()]
