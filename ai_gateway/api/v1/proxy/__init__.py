from fastapi import APIRouter

from ai_gateway.api.v1.proxy import anthropic, vertex_ai

__all__ = [
    "router",
]


router = APIRouter()
router.include_router(anthropic.router)
router.include_router(vertex_ai.router)
