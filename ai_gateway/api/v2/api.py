from fastapi import APIRouter

from ai_gateway.api.v2.endpoints import code
from ai_gateway.api.v2.experimental import code as experimental_code

api_router = APIRouter()
api_router.prefix = "/v2"

api_router.include_router(code.router)
api_router.include_router(experimental_code.router, prefix="/experimental")
