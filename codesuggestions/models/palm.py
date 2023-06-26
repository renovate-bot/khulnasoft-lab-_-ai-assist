from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from codesuggestions.models import TextGenBaseModel, TextGenModelOutput
from codesuggestions.instrumentators.base import TextGenModelInstrumentator

from google.protobuf import json_format, struct_pb2
from google.api_core import gapic_v1, retry as retries

__all__ = [
    "PalmModel",
    "PalmCodeGenBaseModel",
    "PalmCodeBisonModel",
    "PalmCodeGeckoModel",
    "PalmCodeGenModel",
]


class PalmPredictionResponse(ABC):
    predictions: MutableSequence


class PalmPredictionClient(ABC):

    @abstractmethod
    def predict(
        self,
        request: Optional[dict] = None,
        *,
        endpoint: Optional[str] = None,
        instances: Optional[MutableSequence[struct_pb2.Value]] = None,
        parameters: Optional[struct_pb2.Value] = None,
        retry: Union[retries.Retry, object] = gapic_v1.method.DEFAULT,
        timeout: Union[float, object] = gapic_v1.method.DEFAULT,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> PalmPredictionResponse:
        pass


class PalmModel(str, Enum):
    TEXT_BISON = "text-bison"
    CODE_BISON = "code-bison"
    CODE_GECKO = "code-gecko"


class PalmCodeGenBaseModel(TextGenBaseModel):
    # Max number of tokens the model can handle
    MAX_MODEL_LEN = 2048

    # Separator used to version models
    # E.g.: code-bison@001
    SEP_MODEL_VERSION = "@"

    def __init__(
        self,
        model_name: str,
        client: PalmPredictionClient,
        project: str,
        location: str,
        timeout: int = 30
    ):
        self.client = client
        self.timeout = timeout

        self.endpoint = f"projects/{project}/locations/{location}/publishers/google/models/{model_name}"
        self.instrumentator = TextGenModelInstrumentator("vertex-ai", model_name)

    def _generate(
        self,
        input_data: dict,
        temperature: float,
        max_decode_steps: int,
        top_p: float,
        top_k: int
    ) -> Optional[TextGenModelOutput]:
        instance = json_format.ParseDict(input_data, struct_pb2.Value())
        instances = [instance]
        parameters_dict = {"temperature": temperature, "maxDecodeSteps": max_decode_steps, "topP": top_p, "topK": top_k}
        parameters = json_format.ParseDict(parameters_dict, struct_pb2.Value())

        response = self.client.predict(
            endpoint=self.endpoint, instances=instances, parameters=parameters, timeout=self.timeout
        )

        predictions = response.predictions
        for prediction in predictions:
            return TextGenModelOutput(text=prediction.get('content'))

    @staticmethod
    def _resolve_model_version(model_name: str, model_version: str = "latest") -> str:
        return (
            model_name
            if model_version == "latest" or model_version == ""
            else PalmCodeGenBaseModel.SEP_MODEL_VERSION.join([model_name, model_version])
        )

    @abstractmethod
    def generate(
        self,
        prompt: str,
        suffix: str,
        temperature: float = 0.2,
        max_decode_steps: int = 32,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> Optional[TextGenModelOutput]:
        pass


class PalmTextBisonModel(PalmCodeGenBaseModel):
    def __init__(
        self,
        client: PalmPredictionClient,
        project: str,
        location: str,
        version: str = "latest"
    ):
        model_name = PalmCodeGenBaseModel._resolve_model_version(
            PalmModel.TEXT_BISON,
            version,
        )
        super().__init__(model_name, client, project, location)

    def generate(
        self,
        prompt: str,
        suffix: str,
        temperature: float = 0.2,
        max_decode_steps: int = 32,
        top_p: float = 0.95,
        top_k: int = 40
    ) -> Optional[TextGenModelOutput]:
        input_data = {"content": prompt}
        with self.instrumentator.watch(prompt):
            res = self._generate(input_data, temperature, max_decode_steps, top_p, top_k)

        return res


class PalmCodeBisonModel(PalmCodeGenBaseModel):
    def __init__(
        self,
        client: PalmPredictionClient,
        project: str,
        location: str,
        version: str = "latest",
    ):
        model_name = PalmCodeGenBaseModel._resolve_model_version(
            PalmModel.CODE_BISON,
            version,
        )
        super().__init__(model_name, client, project, location)

    def generate(
        self,
        prompt: str,
        suffix: str,
        temperature: float = 0.2,
        max_decode_steps: int = 32,
        top_p: float = 0.95,
        top_k: int = 40
    ) -> Optional[TextGenModelOutput]:
        input_data = {"prefix": prompt}
        with self.instrumentator.watch(prompt):
            res = self._generate(input_data, temperature, max_decode_steps, top_p, top_k)

        return res


class PalmCodeGeckoModel(PalmCodeGenBaseModel):
    def __init__(
        self,
        client: PalmPredictionClient,
        project: str,
        location: str,
        version: str = "latest"
    ):
        model_name = PalmCodeGenBaseModel._resolve_model_version(
            PalmModel.CODE_GECKO,
            version,
        )
        super().__init__(model_name, client, project, location)

    def generate(
        self,
        prompt: str,
        suffix: str,
        temperature: float = 0.2,
        max_decode_steps: int = 32,
        top_p: float = 0.95,
        top_k: int = 40
    ) -> Optional[TextGenModelOutput]:
        input_data = {"prefix": prompt, "suffix": suffix}

        with self.instrumentator.watch(prompt):
            res = self._generate(input_data, temperature, max_decode_steps, top_p, top_k)

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
        client: PalmPredictionClient,
        project: str,
        location: str,
    ) -> PalmCodeGenBaseModel:
        model_name, _, model_version = name.partition(PalmCodeGenBaseModel.SEP_MODEL_VERSION)

        if model := PalmCodeGenModel.models.get(PalmModel(model_name), None):
            return model(client, project, location, version=model_version)

        raise ValueError(f"no model found by the name '{name}'")
