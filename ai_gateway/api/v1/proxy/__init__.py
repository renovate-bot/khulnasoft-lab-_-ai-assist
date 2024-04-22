from fastapi import APIRouter

from ai_gateway.api.v1.proxy import anthropic

__all__ = [
    "router",
]


router = APIRouter()
router.include_router(anthropic.router)
