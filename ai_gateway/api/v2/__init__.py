from fastapi import APIRouter

from ai_gateway.api.v2 import code

__all__ = ["api_router"]

api_router = APIRouter()

# We don't include the `code` prefix here, as we need to support the legacy `/completions` endpoint.
api_router.include_router(code.router, tags=["completions"])
