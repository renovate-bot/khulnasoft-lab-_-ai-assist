from fastapi import APIRouter

from ai_gateway.api.v2 import code

api_router = APIRouter()
api_router.prefix = "/v2"

api_router.include_router(code.router)
