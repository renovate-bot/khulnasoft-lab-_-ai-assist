from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, NamedTuple

import structlog
from anthropic import AsyncAnthropic
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient
from pydantic import BaseModel

from ai_gateway.config import Config
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator

# TODO: The instrumentator needs the config here to know what limit needs to be
# reported for a model. This would be nicer if we dependency inject the instrumentator
# into the model itself
# https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/384
config = Config()

__all__ = [
    "KindModelProvider",
    "ModelAPIError",
    "ModelAPICallError",
    "ModelMetadata",
    "TokensConsumptionMetadata",
    "SafetyAttributes",
    "ModelBase",
    "grpc_connect_vertex",
    "connect_anthropic",
]

log = structlog.stdlib.get_logger("codesuggestions")


class KindModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex-ai"
    LITELLM = "litellm"


class ModelAPIError(Exception):
    def __init__(self, message: str, errors: tuple = (), details: tuple = ()):
        self.message = message
        self._errors = errors
        self._details = details

    def __str__(self):
        message = self.message

        if self.details:
            message = f"{message} {self.details}"

        return message

    @property
    def errors(self) -> list[Any]:
        return list(self._errors)

    @property
    def details(self) -> list[Any]:
        return list(self._details)


class ModelAPICallError(ModelAPIError):
    code: int

    def __init__(self, message: str, errors: tuple = (), details: tuple = ()):
        super().__init__(f"{self.code} {message}", errors=errors, details=details)


class TokensConsumptionMetadata(NamedTuple):
    input_tokens: int
    output_tokens: int


class SafetyAttributes(BaseModel):
    blocked: bool = False
    categories: list[str] = []
    errors: list[int] = []


class ModelMetadata(NamedTuple):
    name: str
    engine: str


class ModelBase(ABC):
    MAX_MODEL_LEN = 2048

    @property
    def instrumentator(self) -> ModelRequestInstrumentator:
        return ModelRequestInstrumentator(
            model_engine=self.metadata.engine,
            model_name=self.metadata.name,
            concurrency_limit=config.model_engine_concurrency_limits.for_model(
                engine=self.metadata.engine, name=self.metadata.name
            ),
        )

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        pass


def grpc_connect_vertex(client_options: dict) -> PredictionServiceAsyncClient:
    log.info("Initializing Vertex AI client", **client_options)
    return PredictionServiceAsyncClient(client_options=client_options)


def connect_anthropic(**kwargs: Any) -> AsyncAnthropic:
    return AsyncAnthropic(**kwargs)
