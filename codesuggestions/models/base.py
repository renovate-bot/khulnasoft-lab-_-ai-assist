from abc import ABC, abstractmethod
from google.oauth2 import service_account
from typing import NamedTuple

import numpy as np
import tritonclient.grpc as triton_grpc_util
import vertexai
from tritonclient.utils import np_to_triton_dtype

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


def grpc_connect_triton(host: str, port: int, verbose: bool = False) -> triton_grpc_util.InferenceServerClient:
    channel_opt = [
        ("grpc.max_send_message_length", triton_grpc_util.MAX_GRPC_MESSAGE_SIZE),
        ("grpc.max_receive_message_length", triton_grpc_util.MAX_GRPC_MESSAGE_SIZE),
        ("grpc.keepalive_time_ms", 30_000),
        ("grpc.keepalive_timeout_ms", 20_000),
        ("grpc.keepalive_permit_without_calls", 2),
        ("grpc.http2.max_pings_without_data", False),
        ("grpc.lb_policy_name", "round_robin"),
    ]

    return triton_grpc_util.InferenceServerClient(
        url=f"{host}:{port}",
        verbose=verbose,
        channel_args=channel_opt,
    )


def vertex_ai_init(project: str, location: str, credential_path: str):
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    vertexai.init(project=project, location=location, credentials=credentials)
