from abc import ABC, abstractmethod

import numpy as np
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as triton_grpc_util

__all__ = [
    "BaseModel",
    "grpc_input_from_np",
    "grpc_requested_output",
    "grpc_connect_triton",
]


class BaseModel(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> str:
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
    return triton_grpc_util.InferenceServerClient(url=f"{host}:{port}", verbose=verbose)
