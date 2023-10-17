from fastapi import APIRouter

from ai_gateway.api.v2.endpoints.chat import evaluation

__all__ = ["router"]

router = APIRouter()

router.include_router(evaluation.router, prefix="/evaluation")
