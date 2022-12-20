from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware, AuthenticationBackend
from starlette.middleware.base import BaseHTTPMiddleware

__all__ = [
    "MiddlewareLogRequest",
    "MiddlewareAuthentication",
]


class MiddlewareLogRequest(Middleware):
    class CustomHeaderMiddleware(BaseHTTPMiddleware):
        # TODO: https://www.starlette.io/middleware/#basehttpmiddleware
        async def dispatch(self, request, call_next):
            print("logging")
            return await call_next(request)

    def __init__(self):
        super().__init__(MiddlewareLogRequest.CustomHeaderMiddleware)


class MiddlewareAuthentication(Middleware):
    class AuthBackend(AuthenticationBackend):
        # TODO: https://www.starlette.io/authentication/
        async def authenticate(self, conn):
            print("authentication")
            return

    def __init__(self):
        super().__init__(AuthenticationMiddleware, backend=MiddlewareAuthentication.AuthBackend())
