import logging
from hashlib import sha1
from typing import Optional

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

log = logging.getLogger("codesuggestions")


class _PathResolver:
    def __init__(self, endpoints: list[str]):
        self.endpoints = set(endpoints if endpoints else [])

    def skip_path(self, path: str) -> bool:
        if path in self.endpoints:
            return True
        return False


class MiddlewareLogRequest(Middleware):
    class CustomHeaderMiddleware(BaseHTTPMiddleware):
        def __init__(self, skip_endpoints: list[str], *args, **kwargs):
            self.path_resolver = _PathResolver(skip_endpoints)
            super().__init__(*args, **kwargs)

        async def dispatch(self, request, call_next):
            if not self.path_resolver.skip_path(request.url.path):
                user = sha1(request.client.host.encode("utf-8")).hexdigest()
                log.info(f"Received request - {user}")
            return await call_next(request)

    def __init__(self, skip_endpoints: Optional[list] = None):
        super().__init__(MiddlewareLogRequest.CustomHeaderMiddleware, skip_endpoints=skip_endpoints)


class MiddlewareAuthentication(Middleware):
    class AuthBackend(AuthenticationBackend):
        PREFIX_BEARER_HEADER = "bearer"
        AUTH_HEADER = "Authorization"

        def __init__(self, auth_provider: AuthProvider, bypass_auth: bool, skip_endpoints: list[str]):
            self.auth_provider = auth_provider
            self.bypass_auth = bypass_auth
            self.path_resolver = _PathResolver(skip_endpoints)

        async def authenticate(self, conn: HTTPConnection):
            """
            Ref: https://www.starlette.io/authentication/
            """

            if self.path_resolver.skip_path(conn.url.path):
                return

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

    def __init__(
        self,
        auth_provider: AuthProvider,
        bypass_auth: bool = False,
        skip_endpoints: Optional[list] = None,
    ):
        super().__init__(
            AuthenticationMiddleware,
            backend=MiddlewareAuthentication.AuthBackend(auth_provider, bypass_auth, skip_endpoints),
            on_error=MiddlewareAuthentication.on_auth_error,
        )
