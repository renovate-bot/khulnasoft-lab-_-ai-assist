from fastapi import APIRouter

from ai_gateway.api.v1.amazon_q import application, events

__all__ = [
    "router",
]

router = APIRouter()
router.include_router(application.router, prefix="/oauth")
router.include_router(events.router)
