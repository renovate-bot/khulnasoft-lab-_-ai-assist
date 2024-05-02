from fastapi import APIRouter

from ai_gateway.api.v1 import chat, search, x_ray

__all__ = ["api_router"]

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(x_ray.router, prefix="/x-ray", tags=["x-ray"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
