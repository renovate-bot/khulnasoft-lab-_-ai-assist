from abc import abstractmethod
from enum import Enum
from http.client import BAD_REQUEST, INTERNAL_SERVER_ERROR
from typing import Optional, Sequence

import structlog
from google.api_core.exceptions import InternalServerError, InvalidArgument
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient, PredictResponse
from google.protobuf import json_format, struct_pb2

from ai_gateway.models import ModelMetadata, TextGenBaseModel, TextGenModelOutput
from ai_gateway.models.base import ModelAPICallError, ModelInput, SafetyAttributes

__all__ = [
    "VertexModelAPICallError",
    "VertexModelInvalidArgument",
    "VertexModelInternalError",
    "PalmModel",
    "PalmCodeGenBaseModel",
    "PalmCodeBisonModel",
    "PalmCodeGeckoModel",
    "PalmCodeGenModel",
]

log = structlog.stdlib.get_logger("codesuggestions")


class VertexModelAPICallError(
    ModelAPICallError,
):
    def __init__(self, message: str, errors: tuple = (), details: tuple = ()):
        message = f"Vertex model API error: {message.lower().strip('.')}"
        super().__init__(message, errors, details)


class VertexModelInvalidArgument(VertexModelAPICallError):
    code = BAD_REQUEST


class VertexModelInternalError(VertexModelAPICallError):
    code = INTERNAL_SERVER_ERROR


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


class PalmModel(str, Enum):
    TEXT_BISON = "text-bison"
    CODE_BISON = "code-bison"
    CODE_GECKO = "code-gecko"


class PalmCodeGenBaseModel(TextGenBaseModel):
    # Max number of tokens the model can handle
    # Source: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#foundation_models
    MAX_MODEL_LEN = 2048
    # If we assume that 4 characters per token, this gives us an upper bound of approximately
    # how many characters should be in the prompt.
    UPPER_BOUND_MODEL_CHARS = MAX_MODEL_LEN * 5

    # Separator used to version models
    # E.g.: code-bison@001
    SEP_MODEL_VERSION = "@"

    _MODEL_ENGINE = "vertex-ai"

    def __init__(
        self,
        model_name: PalmModel,
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        model_version: str,
        timeout: int = 30,
    ):
        self.client = client
        self.timeout = timeout

        model_version = "latest" if model_version == "" else model_version
        model_name = PalmCodeGenBaseModel.SEP_MODEL_VERSION.join(
            [model_name, model_version]
        )

        self._metadata = ModelMetadata(
            name=model_name, engine=PalmCodeGenBaseModel._MODEL_ENGINE
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

        try:
            response = await self.client.predict(
                endpoint=self.endpoint,
                instances=instances,
                parameters=parameters,
                timeout=self.timeout,
            )
            response = PredictResponse.to_dict(response)

            predictions = response.get("predictions", [])
        except InvalidArgument as ex:
            raise VertexModelInvalidArgument(ex.message, errors=(ex,))
        except InternalServerError as ex:
            raise VertexModelInternalError(ex.message, errors=(ex,))

        for prediction in predictions:
            return TextGenModelOutput(
                text=prediction.get("content"),
                score=prediction.get("score"),
                safety_attributes=SafetyAttributes(
                    **prediction.get("safetyAttributes", {})
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
        version: str = "latest",
    ):
        super().__init__(PalmModel.TEXT_BISON, client, project, location, version)

    async def generate(
        self,
        prompt: str,
        suffix: str,
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


class PalmCodeBisonModel(PalmCodeGenBaseModel):
    MAX_MODEL_LEN = 4096

    def __init__(
        self,
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        version: str = "latest",
    ):
        super().__init__(PalmModel.CODE_BISON, client, project, location, version)

    async def generate(
        self,
        prompt: str,
        suffix: str,
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


class PalmCodeGeckoModel(PalmCodeGenBaseModel):
    MAX_MODEL_LEN = 2048
    DEFAULT_STOP_SEQUENCES = ["\n\n"]

    def __init__(
        self,
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
        version: str = "latest",
    ):
        super().__init__(PalmModel.CODE_GECKO, client, project, location, version)

    async def generate(
        self,
        prompt: str,
        suffix: str,
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


class PalmCodeGenModel:
    models = {
        PalmModel.TEXT_BISON: PalmTextBisonModel,
        PalmModel.CODE_BISON: PalmCodeBisonModel,
        PalmModel.CODE_GECKO: PalmCodeGeckoModel,
    }

    @staticmethod
    def from_model_name(
        name: str,
        client: PredictionServiceAsyncClient,
        project: str,
        location: str,
    ) -> PalmCodeGenBaseModel:
        model_name, _, model_version = name.partition(
            PalmCodeGenBaseModel.SEP_MODEL_VERSION
        )

        if model := PalmCodeGenModel.models.get(PalmModel(model_name), None):
            return model(client, project, location, version=model_version)

        raise ValueError(f"no model found by the name '{name}'")
