from abc import ABC, abstractmethod
from tritonclient.grpc import service_pb2_grpc
from typing import NamedTuple

import grpc
import numpy as np
import tritonclient.grpc as triton_grpc_util
import vertexai
from tritonclient.utils import np_to_triton_dtype
from py_grpc_prometheus.prometheus_client_interceptor import PromClientInterceptor

__all__ = [
    "TextGenModelOutput",
    "TextGenBaseModel",
    "grpc_input_from_np",
    "grpc_requested_output",
    "grpc_connect_triton",
    "vertex_ai_init",
]


class TextGenModelOutput(NamedTuple):
    text: str


class TextGenBaseModel(ABC):
    MAX_MODEL_LEN = 1

    @abstractmethod
    def generate(
        self,
        content: str,
        temperature: float = 0.2,
        max_decode_steps: int = 16,
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


def vertex_ai_init(project: str, location: str):
    vertexai.init(project=project, location=location)
