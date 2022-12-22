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
    uvicorn_logger: dict


class AuthConfig(NamedTuple):
    gitlab_api_base_url: str
    bypass: bool


class Config:
    BOOLEAN_STATES = {
        '1': True, 'yes': True, 'true': True, 'on': True,
        '0': False, 'no': False, 'false': False, 'off': False
    }

    UVICORN_LOGGER = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(asctime)s - "%(request_line)s" %(status_code)s',
                "use_colors": True
            },
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": '%(levelprefix)s %(asctime)s - "%(message)s"',
                "use_colors": True,
            },
        },
        "handlers": {
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn.access": {
                "handlers": ["access"],
                # "level": "INFO",
                "propagate": False
            },
            "codesuggestions": {
                "handlers": ["default"],
                "level": "INFO"
            },
        },
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
            uvicorn_logger=Config.UVICORN_LOGGER,
        )

    @property
    def auth(self) -> AuthConfig:
        return AuthConfig(
            gitlab_api_base_url=Config._get_value("GITLAB_API_URL", "https://gitlab.com/api/v4/"),
            bypass=Config._str_to_bool(Config._get_value("AUTH_BYPASS_EXTERNAL", "False"))
        )

    @staticmethod
    def _get_value(value: str, default: Optional[Any]):
        return os.environ.get(value, default)

    @staticmethod
    def _str_to_bool(value: str):
        if value.lower() not in Config.BOOLEAN_STATES:
            raise ValueError('Not a boolean: %s' % value)
        return Config.BOOLEAN_STATES[value.lower()]
