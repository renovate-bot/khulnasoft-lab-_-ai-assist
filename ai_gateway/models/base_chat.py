from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, AsyncIterator, Union

from pydantic import BaseModel, StringConstraints

from ai_gateway.config import Config
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.models.base import ModelMetadata, TextGenModelChunk, TextGenModelOutput

config = Config()


__all__ = ["ChatModelBase", "Role", "Message"]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: Annotated[str, StringConstraints(max_length=400000)]


class ChatModelBase(ABC):
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

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        stream: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 16,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> Union[TextGenModelOutput, AsyncIterator[TextGenModelChunk]]:
        pass
