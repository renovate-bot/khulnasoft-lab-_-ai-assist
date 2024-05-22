from fastapi import APIRouter

from ai_gateway.api.v1.code import user_access_token

__all__ = [
    "router",
]

router = APIRouter()

router.include_router(user_access_token.router)
