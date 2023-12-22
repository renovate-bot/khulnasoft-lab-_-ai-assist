from fastapi import APIRouter

from ai_gateway.api.v1 import chat
from ai_gateway.api.v1 import x_ray

api_router = APIRouter()
api_router.prefix = "/v1"

api_router.include_router(chat.router, prefix="/chat")
api_router.include_router(x_ray.router, prefix="/x-ray")
