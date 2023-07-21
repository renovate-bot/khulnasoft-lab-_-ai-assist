from abc import ABC, abstractmethod
from typing import NamedTuple, Any, Optional

import grpc
import numpy as np
import tritonclient.grpc as triton_grpc_util
from tritonclient.grpc import service_pb2_grpc
from tritonclient.utils import np_to_triton_dtype
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient
from py_grpc_prometheus.prometheus_client_interceptor import PromClientInterceptor

__all__ = [
    "ModelAPICallError",

    "TextGenModelOutput",
    "TextGenBaseModel",
    "grpc_input_from_np",
    "grpc_requested_output",
    "grpc_connect_triton",
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


def grpc_input_from_np(name: str, data: np.ndarray) -> triton_grpc_util.InferInput:
    t = triton_grpc_util.InferInput(
        name, list(data.shape), np_to_triton_dtype(data.dtype)
    )
    t.set_data_from_numpy(data)
    return t


def grpc_requested_output(name: str) -> triton_grpc_util.InferRequestedOutput:
    return triton_grpc_util.InferRequestedOutput(name)


def grpc_connect_triton(
        host: str,
        port: int,
        interceptor: PromClientInterceptor,
        verbose: bool = False
) -> triton_grpc_util.InferenceServerClient:
    # These settings MUST be kept in sync with the Triton server config:
    # https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
    channel_opt = [
        ("grpc.max_send_message_length", triton_grpc_util.MAX_GRPC_MESSAGE_SIZE),
        ("grpc.max_receive_message_length", triton_grpc_util.MAX_GRPC_MESSAGE_SIZE),
        ("grpc.lb_policy_name", "round_robin"),
    ]

    client = triton_grpc_util.InferenceServerClient(
        url=f"{host}:{port}",
        verbose=verbose,
        channel_args=channel_opt,
    )
    # Unfortunately we have to reach into the internals of the client in order
    # to instrument it. If these internals ever change, GRPC client instrumentation
    # may no longer work.
    client._channel = grpc.intercept_channel(client._channel, interceptor)
    client._client_stub = service_pb2_grpc.GRPCInferenceServiceStub(client._channel)

    return client


def grpc_connect_vertex(client_options: dict) -> PredictionServiceAsyncClient:
    return PredictionServiceAsyncClient(client_options=client_options)
