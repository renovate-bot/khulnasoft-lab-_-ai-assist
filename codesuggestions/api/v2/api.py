from fastapi import APIRouter

from codesuggestions.api.v2.endpoints import suggestions

api_router = APIRouter()
api_router.prefix = "/v2"

api_router.include_router(suggestions.router)
