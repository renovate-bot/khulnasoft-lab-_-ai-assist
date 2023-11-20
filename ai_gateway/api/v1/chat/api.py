import os

from fastapi import APIRouter

from ai_gateway.api.v1.chat import agent, evaluation

__all__ = ["router"]

router = APIRouter()

router.include_router(agent.router)

# Hack: Avoid deploying the chat evaluation endpoints to production
# TODO: Protect the API endpoints verifying the token scopes
if os.environ.get("F_CHAT_EVALUATION_API", "0").lower() in ("1", "true"):
    router.include_router(evaluation.create_router(), prefix="/evaluation")
