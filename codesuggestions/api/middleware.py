import logging as log

from starlette.middleware import Middleware
from starlette.authentication import AuthenticationError, HTTPConnection
from starlette.middleware.authentication import AuthenticationMiddleware, AuthenticationBackend
from starlette.middleware.base import BaseHTTPMiddleware, Request
from starlette.responses import PlainTextResponse

from codesuggestions.auth import AuthProvider

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
        PREFIX_BEARER_HEADER = "bearer"
        AUTH_HEADER = "Authorization"

        def __init__(self, auth_provider: AuthProvider, bypass_auth: bool):
            self.auth_provider = auth_provider
            self.bypass_auth = bypass_auth

        async def authenticate(self, conn: HTTPConnection):
            """
            Ref: https://www.starlette.io/authentication/
            """

            if self.bypass_auth:
                log.critical("Auth is disabled, all users allowed")
                return

            if self.AUTH_HEADER not in conn.headers:
                raise AuthenticationError("No authorization header presented")

            header = conn.headers[self.AUTH_HEADER]
            bearer, _, token = header.partition(" ")
            if bearer.lower() != self.PREFIX_BEARER_HEADER:
                raise AuthenticationError('Invalid authorization header')

            is_auth = self.auth_provider.authenticate(token)
            if not is_auth:
                raise AuthenticationError("Forbidden by auth provider")

            return

    @staticmethod
    def on_auth_error(_: Request, e: Exception):
        log.error(e)
        return PlainTextResponse(status_code=401)

    def __init__(self, auth_provider: AuthProvider, bypass_auth: bool = False, **kwargs):
        super().__init__(
            AuthenticationMiddleware,
            backend=MiddlewareAuthentication.AuthBackend(auth_provider, bypass_auth),
            on_error=MiddlewareAuthentication.on_auth_error,
            **kwargs,
        )
