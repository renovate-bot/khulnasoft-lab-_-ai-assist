from fastapi import FastAPI
from dependency_injector.wiring import Provide, inject

from codesuggestions.api.suggestions import router as http_router_suggestions
from codesuggestions.api.middleware import MiddlewareAuthentication, MiddlewareLogRequest
from codesuggestions.deps import FastApiContainer

__all__ = [
    "create_fast_api_server",
]


@inject
def create_fast_api_server(
        config: dict = Provide[FastApiContainer.config.fastapi],
        auth_middleware: MiddlewareAuthentication = Provide[FastApiContainer.auth_middleware],
        log_middleware: MiddlewareLogRequest = Provide[FastApiContainer.log_middleware],
):
    fastapi_app = FastAPI(
        title="GitLab Code Suggestions",
        description="GitLab Code Suggestions API to serve code completion predictions",
        openapi_url=config["openapi_url"],
        docs_url=config["docs_url"],
        redoc_url=config["redoc_url"],
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        middleware=[
            auth_middleware,
            log_middleware,
        ],
    )

    fastapi_app.include_router(http_router_suggestions, prefix="/v1")

    return fastapi_app
