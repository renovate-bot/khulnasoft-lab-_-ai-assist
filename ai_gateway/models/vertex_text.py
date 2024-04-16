from abc import abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Union

import structlog
from google.api_core.exceptions import GoogleAPICallError, GoogleAPIError
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient, PredictResponse
from google.protobuf import json_format, struct_pb2

from ai_gateway.models.base import (
    KindModelProvider,
    ModelAPICallError,
    ModelAPIError,
    ModelInput,
    ModelMetadata,
    SafetyAttributes,
    TextGenBaseModel,
    TextGenModelOutput,
    TokensConsumptionMetadata,
)

__all__ = [
    "PalmCodeBisonModel",
    "PalmTextBisonModel",
    "PalmCodeGeckoModel",
    "PalmCodeGenBaseModel",
    "KindVertexTextModel",
    "VertexAPIConnectionError",
    "VertexAPIStatusError",
]

log = structlog.stdlib.get_logger("codesuggestions")


class VertexAPIConnectionError(ModelAPIError):
    @classmethod
    def from_exception(cls, ex: GoogleAPIError):
        message = f"Vertex Model API error: {type(ex).__name__}"

        if hasattr(ex, "message"):
            message = f"{message} {ex.message}"

        return cls(message, errors=(ex,))


class VertexAPIStatusError(ModelAPICallError):
    @classmethod
    def from_exception(cls, ex: GoogleAPICallError):
        cls.code = ex.code
        message = f"Vertex Model API error: {type(ex).__name__} {ex.message}"

        return cls(message, errors=(ex,), details=ex.details)


class CodeBisonModelInput(ModelInput):
    def __init__(self, prefix):
        self.prefix = prefix

    def is_valid(self) -> bool:
        return len(self.prefix) > 0

    def dict(self) -> dict:
        return {"prefix": self.prefix}


class TextBisonModelInput(ModelInput):
    def __init__(self, prefix):
        self.prefix = prefix

    def is_valid(self) -> bool:
        return len(self.prefix) > 0

    def dict(self) -> dict:
        return {"content": self.prefix}


class CodeGeckoModelInput(ModelInput):
    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix

    def is_valid(self) -> bool:
        return len(self.prefix) > 0

    def dict(self) -> dict:
        return {"prefix": self.prefix, "suffix": self.suffix}


class KindVertexTextModel(str, Enum):
    # Avoid using model versions that only specify the major version number
    # similar to `KindAnthropicModel`.
    CODE_BISON_002 = "code-bison@002"
    CODE_GECKO_002 = "code-gecko@002"
    TEXT_BISON_002 = "text-bison@002"


class PalmCodeGenBaseModel(TextGenBaseModel):
    # Max number of tokens the model can handle
    # Source: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#foundation_models
    MAX_MODEL_LEN = 2048
    # If we assume that 4 characters per token, this gives us an upper bound of approximately
    # how many characters should be in the prompt.
    UPPER_BOUND_MODEL_CHARS = MAX_MODEL_LEN * 5

    def __init__(
        self,
        model_name: str,
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        timeout: int = 30,
    ):
        self.client = client
        self.timeout = timeout

        self._metadata = ModelMetadata(
            name=model_name, engine=KindModelProvider.VERTEX_AI.value
        )
        self.endpoint = f"projects/{project}/locations/{location}/publishers/google/models/{model_name}"

    async def _generate(
        self,
        input: ModelInput,
        temperature: float,
        max_output_tokens: int,
        top_p: float,
        top_k: int,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> Optional[TextGenModelOutput]:
        if not input.is_valid():
            return TextGenModelOutput(
                text="", score=0, safety_attributes=SafetyAttributes()
            )

        input_data = input.dict()

        instance = json_format.ParseDict(input_data, struct_pb2.Value())
        instances = [instance]
        parameters_dict = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": top_p,
            "topK": top_k,
        }
        if stop_sequences:
            parameters_dict["stopSequences"] = stop_sequences

        parameters = json_format.ParseDict(parameters_dict, struct_pb2.Value())

        log.debug("codegen vertex call:", input=input_data, parameters=parameters_dict)

        with self.instrumentator.watch():
            try:
                response = await self.client.predict(
                    endpoint=self.endpoint,
                    instances=instances,
                    parameters=parameters,
                    timeout=self.timeout,
                )
                response = PredictResponse.to_dict(response)
                tokens_metatada = response.get("metadata", {}).get("tokenMetadata", {})
                log.debug(
                    "codegen vertex response:",
                    tokens_metatada=tokens_metatada,
                    input_tokens=tokens_metatada.get("inputTokenCount", {}).get(
                        "totalTokens", None
                    ),
                    output_tokens=tokens_metatada.get("outputTokenCount", {}).get(
                        "totalTokens", None
                    ),
                )
                predictions = response.get("predictions", [])
            except GoogleAPICallError as ex:
                raise VertexAPIStatusError.from_exception(ex)
            except GoogleAPIError as ex:
                raise VertexAPIConnectionError.from_exception(ex)

        for prediction in predictions:
            return TextGenModelOutput(
                text=prediction.get("content"),
                score=prediction.get("score"),
                safety_attributes=SafetyAttributes(
                    **prediction.get("safetyAttributes", {})
                ),
                metadata=TokensConsumptionMetadata(
                    input_tokens=tokens_metatada.get("inputTokenCount", {}).get(
                        "totalTokens", None
                    ),
                    output_tokens=tokens_metatada.get("outputTokenCount", {}).get(
                        "totalTokens", None
                    ),
                ),
            )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        suffix: str,
        stream: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 32,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> Optional[TextGenModelOutput]:
        pass


class PalmTextBisonModel(PalmCodeGenBaseModel):
    MAX_MODEL_LEN = 8192

    def __init__(
        self,
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        model_name: str = KindVertexTextModel.TEXT_BISON_002.value,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(model_name, client, project, location, *args, **kwargs)

    async def generate(
        self,
        prompt: str,
        suffix: str,
        stream: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 32,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> Optional[TextGenModelOutput]:
        model_input = TextBisonModelInput(prompt)
        res = await self._generate(
            model_input,
            temperature,
            max_output_tokens,
            top_p,
            top_k,
            stop_sequences,
        )

        return res

    @classmethod
    def from_model_name(
        cls,
        name: Union[str, KindVertexTextModel],
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        **kwargs: Any,
    ):
        name = _resolve_model_name(name, "text-bison")

        return cls(client, project, location, model_name=name.value, **kwargs)


class PalmCodeBisonModel(PalmCodeGenBaseModel):
    MAX_MODEL_LEN = 4096

    def __init__(
        self,
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        model_name: str = KindVertexTextModel.CODE_BISON_002.value,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(model_name, client, project, location, *args, **kwargs)

    async def generate(
        self,
        prompt: str,
        suffix: str,
        stream: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> Optional[TextGenModelOutput]:
        model_input = CodeBisonModelInput(prompt)
        res = await self._generate(
            model_input,
            temperature,
            max_output_tokens,
            top_p,
            top_k,
            stop_sequences,
        )

        return res

    @classmethod
    def from_model_name(
        cls,
        name: Union[str, KindVertexTextModel],
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        **kwargs: Any,
    ):
        name = _resolve_model_name(name, "code-bison")

        return cls(client, project, location, model_name=name.value, **kwargs)


class PalmCodeGeckoModel(PalmCodeGenBaseModel):
    MAX_MODEL_LEN = 2048
    DEFAULT_STOP_SEQUENCES = ["\n\n"]
    PREFIX_MODEL_IDENTIFIER = "code-gecko"

    def __init__(
        self,
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        model_name: str = KindVertexTextModel.CODE_GECKO_002.value,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(model_name, client, project, location, *args, **kwargs)

    async def generate(
        self,
        prompt: str,
        suffix: str,
        stream: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 64,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> Optional[TextGenModelOutput]:
        model_input = CodeGeckoModelInput(prompt, suffix)

        if not stop_sequences:
            stop_sequences = PalmCodeGeckoModel.DEFAULT_STOP_SEQUENCES

        res = await self._generate(
            model_input,
            temperature,
            max_output_tokens,
            top_p,
            top_k,
            stop_sequences,
        )

        return res

    @classmethod
    def from_model_name(
        cls,
        name: Union[str, KindVertexTextModel],
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        **kwargs: Any,
    ):
        name = _resolve_model_name(name, "code-gecko")

        return cls(client, project, location, model_name=name.value, **kwargs)


def _resolve_model_name(
    name: Union[str, KindVertexTextModel], prefix: str
) -> KindVertexTextModel:
    try:
        name = KindVertexTextModel(name)
    except ValueError:
        raise ValueError(f"no model found by '{name}'")

    if not name.value.startswith(prefix):
        raise ValueError(f"no model found by '{name.value}'")

    return name
