from fastapi import APIRouter
from fastapi_health import health

__all__ = [
    "router",
]

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
)


router.add_api_route("/healthz", health([]))
