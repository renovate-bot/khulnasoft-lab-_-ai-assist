from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Optional

from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient

__all__ = [
    "ModelAPICallError",
    "TextGenModelOutput",
    "TextGenBaseModel",
    "grpc_connect_vertex",
]


class ModelAPICallError(Exception):
    code: Optional[int] = None

    def __init__(self, message: str, errors: tuple = (), details: tuple = ()):
        self.message = message
        self._errors = errors
        self._details = details

    def __str__(self):
        message = f"{self.code} {self.message}"
        if self.details:
            message = f"{message} {', '.join(self.details)}"

        return message

    @property
    def errors(self) -> list[Any]:
        return list(self._errors)

    @property
    def details(self) -> list[Any]:
        return list(self._details)


class ModelInput(ABC):
    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @abstractmethod
    def dict(self) -> dict:
        pass

    def __eq__(self, obj):
        return self.dict() == obj.dict()


class TextGenModelOutput(NamedTuple):
    text: str
    score: float


class TextGenBaseModel(ABC):
    MAX_MODEL_LEN = 1

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_engine(self) -> str:
        pass

    @abstractmethod
    def generate(
        self,
        prefix: str,
        suffix: str,
        temperature: float = 0.2,
        max_output_tokens: int = 16,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> TextGenModelOutput:
        pass


def grpc_connect_vertex(client_options: dict) -> PredictionServiceAsyncClient:
    return PredictionServiceAsyncClient(client_options=client_options)
