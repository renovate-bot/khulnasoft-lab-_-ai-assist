from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, NamedTuple, Union

from anthropic import AsyncAnthropic
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient
from pydantic import BaseModel

from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator

__all__ = [
    "UseCases",
    "ModelProviders",
    "AnthropicModels",
    "VertexModels",
    "PROVIDERS_MODELS_MAP",
    "USE_CASES_MODELS_MAP",
    "ModelAPIError",
    "ModelAPICallError",
    "ModelMetadata",
    "SafetyAttributes",
    "TextGenModelOutput",
    "TextGenModelChunk",
    "TextGenBaseModel",
    "grpc_connect_vertex",
    "connect_anthropic",
]


class UseCases(str, Enum):
    CODE_COMPLETIONS = "code completions"
    CODE_GENERATIONS = "code generations"


class ModelProviders(str, Enum):
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex-ai"


class AnthropicModels(str, Enum):
    CLAUDE_INSTANT_1 = "claude-instant-1"
    CLAUDE_INSTANT_1_1 = "claude-instant-1.1"
    CLAUDE_INSTANT_1_2 = "claude-instant-1.2"
    CLAUDE_2 = "claude-2"
    CLAUDE_2_0 = "claude-2.0"
    CLAUDE_2_1 = "claude-2.1"


class VertexModels(str, Enum):
    CODE_BISON = "code-bison"
    CODE_BISON_002 = "code-bison@002"
    CODE_GECKO = "code-gecko"
    CODE_GECKO_002 = "code-gecko@002"
    TEXT_BISON = "text-bison"
    TEXT_BISON_002 = "text-bison@002"


PROVIDERS_MODELS_MAP = {
    ModelProviders.ANTHROPIC: AnthropicModels,
    ModelProviders.VERTEX_AI: VertexModels,
}

USE_CASES_MODELS_MAP = {
    UseCases.CODE_COMPLETIONS: {
        AnthropicModels.CLAUDE_INSTANT_1,
        AnthropicModels.CLAUDE_INSTANT_1_1,
        AnthropicModels.CLAUDE_INSTANT_1_2,
        VertexModels.CODE_GECKO,
        VertexModels.CODE_GECKO_002,
    },
    UseCases.CODE_GENERATIONS: {
        AnthropicModels.CLAUDE_2,
        AnthropicModels.CLAUDE_2_0,
        AnthropicModels.CLAUDE_2_1,
        VertexModels.CODE_BISON,
        VertexModels.CODE_BISON_002,
    },
}


class ModelAPIError(Exception):
    def __init__(self, message: str, errors: tuple = (), details: tuple = ()):
        self.message = message
        self._errors = errors
        self._details = details

    def __str__(self):
        message = self.message

        if self.details:
            message = f"{message} {', '.join(self.details)}"

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


class ModelMetadata(NamedTuple):
    name: str
    engine: str


class ModelInput(ABC):
    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @abstractmethod
    def dict(self) -> dict:
        pass

    def __eq__(self, obj):
        return self.dict() == obj.dict()


class SafetyAttributes(BaseModel):
    blocked: bool = False
    categories: list[str] = []
    errors: list[int] = []


class TextGenModelOutput(NamedTuple):
    text: str
    score: float
    safety_attributes: SafetyAttributes


class TextGenModelChunk(NamedTuple):
    text: str


class TextGenBaseModel(ABC):
    MAX_MODEL_LEN = 2048

    @property
    def instrumentator(self) -> ModelRequestInstrumentator:
        return ModelRequestInstrumentator(
            model_engine=self.metadata.engine, model_name=self.metadata.name
        )

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        pass

    @abstractmethod
    async def generate(
        self,
        prefix: str,
        suffix: str,
        stream: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 16,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> Union[TextGenModelOutput, AsyncIterator[TextGenModelChunk]]:
        pass


def grpc_connect_vertex(client_options: dict) -> PredictionServiceAsyncClient:
    return PredictionServiceAsyncClient(client_options=client_options)


def connect_anthropic(**kwargs: Any) -> AsyncAnthropic:
    return AsyncAnthropic(**kwargs)
