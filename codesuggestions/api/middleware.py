import logging
import structlog
import time

from asgi_correlation_id.context import correlation_id

from typing import Optional

from fastapi import Response
from fastapi.encoders import jsonable_encoder

from starlette.middleware import Middleware
from starlette.authentication import AuthenticationError, HTTPConnection
from starlette.concurrency import iterate_in_threadpool
from starlette.middleware.authentication import AuthenticationMiddleware, AuthenticationBackend
from starlette.middleware.base import BaseHTTPMiddleware, Request
from starlette.responses import JSONResponse

from uvicorn.protocols.utils import get_path_with_query_string

from codesuggestions.auth import AuthProvider
from codesuggestions.api.timing import timing
from starlette_context import context

__all__ = [
    "MiddlewareLogRequest",
    "MiddlewareAuthentication",
]

log = logging.getLogger("codesuggestions")
access_logger = structlog.stdlib.get_logger("api.access")


class _PathResolver:
    def __init__(self, endpoints: list[str]):
        self.endpoints = set(endpoints)

    @classmethod
    def from_optional_list(cls, endpoints: Optional[list] = None) -> "_PathResolver":
        if endpoints is None:
            endpoints = []
        return cls(endpoints)

    def skip_path(self, path: str) -> bool:
        return path in self.endpoints


class MiddlewareLogRequest(Middleware):
    class CustomHeaderMiddleware(BaseHTTPMiddleware):
        def __init__(self, path_resolver: _PathResolver, *args, **kwargs):
            self.path_resolver = path_resolver
            super().__init__(*args, **kwargs)

        async def dispatch(self, request, call_next):
            if self.path_resolver.skip_path(request.url.path):
                return await call_next(request)

            structlog.contextvars.clear_contextvars()
            # These context vars will be added to all log entries emitted during the request
            request_id = correlation_id.get()
            structlog.contextvars.bind_contextvars(correlation_id=request_id)

            start_time = time.perf_counter_ns()
            # If the call_next raises an error, we still want to return our own 500 response,
            # so we can add headers to it (process time, request ID...)
            response = Response(status_code=500)
            try:
                response = await call_next(request)
            except Exception:
                # TODO: Validate that we don't swallow exceptions (unit test?)
                structlog.stdlib.get_logger("api.error").exception("Uncaught exception")
                raise
            finally:
                process_time = time.perf_counter_ns() - start_time
                status_code = response.status_code
                url = get_path_with_query_string(request.scope)
                client_host = request.client.host
                client_port = request.client.port
                http_method = request.method
                http_version = request.scope["http_version"]
                process_time_s = process_time / 1e9

                if 400 <= status_code < 500:
                    # StreamingResponse is received from the MiddlewareAuthentication, so
                    # we need to read the response ourselves.
                    response_body = [section async for section in response.body_iterator]
                    response.body_iterator = iterate_in_threadpool(iter(response_body))
                    structlog.contextvars.bind_contextvars(response_body=response_body[0].decode())

                fields = dict(
                    url=str(request.url),
                    path=url,
                    status_code=status_code,
                    method=http_method,
                    correlation_id=request_id,
                    http_version=http_version,
                    client_ip=client_host,
                    client_port=client_port,
                    duration_s=process_time_s,
                    user_agent=request.headers.get('User-Agent')
                )
                fields.update(context.data)

                # Recreate the Uvicorn access log format, but add all parameters as structured information
                access_logger.info(
                    f"""{client_host}:{client_port} - "{http_method} {url} HTTP/{http_version}" {status_code}""",
                    **fields)
                response.headers["X-Process-Time"] = str(process_time_s)
                return response

    def __init__(self, skip_endpoints: Optional[list] = None):
        path_resolver = _PathResolver.from_optional_list(skip_endpoints)

        super().__init__(MiddlewareLogRequest.CustomHeaderMiddleware, path_resolver=path_resolver)


class MiddlewareAuthentication(Middleware):
    class AuthBackend(AuthenticationBackend):
        PREFIX_BEARER_HEADER = "bearer"
        AUTH_HEADER = "Authorization"

        def __init__(self, auth_provider: AuthProvider, bypass_auth: bool, path_resolver: _PathResolver):
            self.auth_provider = auth_provider
            self.bypass_auth = bypass_auth
            self.path_resolver = path_resolver

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

            self.authenticate_with_token(token)

        @timing("auth_duration_s")
        def authenticate_with_token(self, token):
            is_auth = self.auth_provider.authenticate(token)
            if not is_auth:
                raise AuthenticationError("Forbidden by auth provider")

            return

    @staticmethod
    def on_auth_error(_: Request, e: Exception):
        content = jsonable_encoder({'error': str(e)})
        return JSONResponse(status_code=401, content=content)

    def __init__(
        self,
        auth_provider: AuthProvider,
        bypass_auth: bool = False,
        skip_endpoints: Optional[list] = None,
    ):
        path_resolver = _PathResolver.from_optional_list(skip_endpoints)

        super().__init__(
            AuthenticationMiddleware,
            backend=MiddlewareAuthentication.AuthBackend(auth_provider, bypass_auth, path_resolver),
            on_error=MiddlewareAuthentication.on_auth_error,
        )
