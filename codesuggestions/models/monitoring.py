import logging

import grpc
import tritonclient.grpc as triton_grpc_util

__all__ = [
    "is_triton_server_live",
]

log = logging.getLogger("codesuggestions")


def is_triton_server_live(
    client: triton_grpc_util.InferenceServerClient,
    timeout: int = 1,
    verbose: bool = False,
):
    try:
        request = triton_grpc_util.service_pb2.ServerLiveRequest()
        if verbose:
            log.info(f"grpc call `is_triton_server_live`: {request=}")

        # noinspection PyProtectedMember
        response = client._client_stub.ServerLive(request=request, timeout=timeout)
        if verbose:
            log.info(f"grpc call `is_triton_server_live`: {response=}")

        return response.live
    except grpc.RpcError as rpc_error:
        triton_grpc_util.raise_error_grpc(rpc_error)
