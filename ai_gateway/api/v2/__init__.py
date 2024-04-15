import os

from dotenv import load_dotenv
from fastapi import APIRouter

from ai_gateway.api.v2 import chat, code

__all__ = ["api_router"]

api_router = APIRouter()

# We don't include the `code` prefix here, as we need to support the legacy `/completions` endpoint.
api_router.include_router(code.router, tags=["completions"])

load_dotenv()
if os.environ.get("AIGW_DUOCHAT_EXPERIMENTAL", False):
    # TODO: Remove this env variable once the agent executor is accepted for prod.
    api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
