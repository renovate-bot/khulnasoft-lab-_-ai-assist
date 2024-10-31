import json
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, NamedTuple, Optional

import httpx
import structlog
from anthropic import AsyncAnthropic
from anthropic._base_client import _DefaultAsyncHttpxClient
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient
from pydantic import BaseModel

from ai_gateway.config import Config
from ai_gateway.feature_flags import FeatureFlag, is_feature_enabled
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.structured_logging import get_request_logger

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
    "init_anthropic_client",
]

log = structlog.stdlib.get_logger("models")
request_log = get_request_logger("models")


class KindModelProvider(StrEnum):
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex-ai"
    LITELLM = "litellm"
    MISTRALAI = "codestral"


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


class TokensConsumptionMetadata(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # number of tokens sent to AI Gateway
    context_tokens_sent: Optional[int] = None
    # number of tokens from context used in the prompt
    context_tokens_used: Optional[int] = None


class SafetyAttributes(BaseModel):
    blocked: bool = False
    categories: list[str] = []
    errors: list[int] = []


class ModelMetadata(NamedTuple):
    name: str
    engine: str
    endpoint: str = None
    api_key: str = None
    identifier: str = None


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

    def name_with_provider(self):
        return (
            self.metadata.identifier if self.metadata.identifier else self.metadata.name
        )


def grpc_connect_vertex(client_options: dict) -> PredictionServiceAsyncClient:
    log.info("Initializing Vertex AI client", **client_options)
    return PredictionServiceAsyncClient(client_options=client_options)


async def log_request(request: httpx.Request):
    if is_feature_enabled(FeatureFlag.EXPANDED_AI_LOGGING):
        try:
            request_content_json = json.loads(request.content.decode("utf8"))
        except Exception:
            request_content_json = {}

        request_log.info(
            "Request to LLM",
            source=__name__,
            request_method=request.method,
            request_url=request.url,
            request_content_json=request_content_json,
        )


def connect_anthropic(**kwargs: Any) -> AsyncAnthropic:
    # Setting 30 seconds to the keep-alive expiry to avoid TLS handshake on every request.
    # See https://www.python-httpx.org/advanced/resource-limits/ for more information.
    limits: httpx.Limits = httpx.Limits(
        max_connections=1000, max_keepalive_connections=100, keepalive_expiry=30
    )

    http_client: httpx.AsyncClient = _DefaultAsyncHttpxClient(
        limits=limits, event_hooks={"request": [log_request]}
    )

    return AsyncAnthropic(http_client=http_client, **kwargs)


def init_anthropic_client(
    mock_model_responses: bool,
) -> AsyncAnthropic | None:
    if mock_model_responses:
        return None

    return connect_anthropic()
