from fastapi import APIRouter

from ai_gateway.api.v1.search import docs

__all__ = [
    "router",
]

router = APIRouter()
router.include_router(docs.router)
