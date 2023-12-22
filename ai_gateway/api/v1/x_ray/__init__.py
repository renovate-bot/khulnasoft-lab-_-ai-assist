from fastapi import APIRouter

from ai_gateway.api.v1.x_ray import libraries

__all__ = [
    "router",
]


router = APIRouter()

# Please, include your sub-routes here to have a single `api_router` exposed.
#
# Example:
# ```python
# router.include_router(agent.router)
# router.include_router(tool_calculator.router)
# ```

router.include_router(libraries.router)
