from fastapi import APIRouter, Depends
from dependency_injector.wiring import Provide, inject

from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.models import monitoring

import tritonclient.grpc as triton_grpc_util
from fastapi_health import health

__all__ = [
    "router",
]

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
)


@inject
def is_triton_server_live(
    grpc_client: triton_grpc_util.InferenceServerClient = Depends(Provide[CodeSuggestionsContainer.grpc_client])
):
    try:
        return monitoring.is_triton_server_live(grpc_client)
    except triton_grpc_util.InferenceServerException:
        return False


router.add_api_route("/tritonz", health([
    is_triton_server_live
]))

router.add_api_route("/healthz", health([]))
