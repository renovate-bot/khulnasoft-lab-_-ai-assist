import logging
import time
import traceback
from typing import Optional, Tuple

import structlog
from asgi_correlation_id.context import correlation_id
from fastapi import Response
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from starlette.authentication import (
    AuthCredentials,
    AuthenticationError,
    HTTPConnection,
)
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.middleware.authentication import (
    AuthenticationBackend,
    AuthenticationMiddleware,
)
from starlette.middleware.base import BaseHTTPMiddleware, Request
from starlette.responses import JSONResponse
from starlette_context import context
from uvicorn.protocols.utils import get_path_with_query_string

from ai_gateway.api.timing import timing
from ai_gateway.auth import AuthProvider, UserClaims
from ai_gateway.auth.user import GitLabUser
from ai_gateway.instrumentators.base import Telemetry, TelemetryInstrumentator
from ai_gateway.tracking.errors import log_exception

__all__ = [
    "MiddlewareLogRequest",
    "MiddlewareAuthentication",
    "MiddlewareModelTelemetry",
]


log = logging.getLogger("codesuggestions")
access_logger = structlog.stdlib.get_logger("api.access")

X_GITLAB_REALM_HEADER = "X-Gitlab-Realm"
X_GITLAB_INSTANCE_ID_HEADER = "X-Gitlab-Instance-Id"
X_GITLAB_GLOBAL_USER_ID_HEADER = "X-Gitlab-Global-User-Id"
X_GITLAB_HOST_NAME_HEADER = "X-Gitlab-Host-Name"
X_GITLAB_VERSION_HEADER = "X-Gitlab-Version"
X_GITLAB_SAAS_NAMESPACE_IDS_HEADER = "X-Gitlab-Saas-Namespace-Ids"
X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER = "X-Gitlab-Saas-Duo-Pro-Namespace-Ids"
X_GITLAB_MODEL_GATEWAY_REQUEST_SENT_AT = "X-Gitlab-Rails-Send-Start"


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

            start_time_total = time.perf_counter()
            start_time_cpu = time.process_time()
            # duration_request represents latency added by sending request from Rails to AI gateway
            try:
                wait_duration = time.time() - float(
                    request.headers.get(X_GITLAB_MODEL_GATEWAY_REQUEST_SENT_AT)
                )
            except (ValueError, TypeError):
                wait_duration = -1

            # If the call_next raises an error, we still want to return our own 500 response,
            # so we can add headers to it (process time, request ID...)
            response = Response(status_code=500)
            try:
                response = await call_next(request)
            except Exception as e:
                # TODO: Validate that we don't swallow exceptions (unit test?)
                context.data["exception"] = {
                    "message": str(e),
                    "backtrace": traceback.format_exc(),
                }
                log_exception(e)
                raise
            finally:
                elapsed_time = time.perf_counter() - start_time_total
                cpu_time = time.process_time() - start_time_cpu
                status_code = response.status_code
                url = get_path_with_query_string(request.scope)
                client_host = request.client.host
                client_port = request.client.port
                http_method = request.method
                http_version = request.scope["http_version"]

                fields = dict(
                    url=str(request.url),
                    path=url,
                    status_code=status_code,
                    method=http_method,
                    correlation_id=request_id,
                    http_version=http_version,
                    client_ip=client_host,
                    client_port=client_port,
                    duration_s=elapsed_time,
                    duration_request=wait_duration,
                    cpu_s=cpu_time,
                    user_agent=request.headers.get("User-Agent"),
                    gitlab_instance_id=request.headers.get(X_GITLAB_INSTANCE_ID_HEADER),
                    gitlab_global_user_id=request.headers.get(
                        X_GITLAB_GLOBAL_USER_ID_HEADER
                    ),
                    gitlab_host_name=request.headers.get(X_GITLAB_HOST_NAME_HEADER),
                    gitlab_version=request.headers.get(X_GITLAB_VERSION_HEADER),
                    gitlab_saas_duo_pro_namespace_ids=request.headers.get(
                        X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER
                    ),
                    gitlab_saas_namespace_ids=request.headers.get(
                        X_GITLAB_SAAS_NAMESPACE_IDS_HEADER
                    ),
                    gitlab_realm=request.headers.get(X_GITLAB_REALM_HEADER),
                )
                fields.update(context.data)

                # Recreate the Uvicorn access log format, but add all parameters as structured information
                access_logger.info(
                    f"""{client_host}:{client_port} - "{http_method} {url} HTTP/{http_version}" {status_code}""",
                    **fields,
                )
                response.headers["X-Process-Time"] = str(elapsed_time)
                return response

    def __init__(self, skip_endpoints: Optional[list] = None):
        path_resolver = _PathResolver.from_optional_list(skip_endpoints)

        super().__init__(
            MiddlewareLogRequest.CustomHeaderMiddleware, path_resolver=path_resolver
        )


class MiddlewareAuthentication(Middleware):
    class AuthBackend(AuthenticationBackend):
        PREFIX_BEARER_HEADER = "bearer"
        AUTH_HEADER = "Authorization"
        AUTH_TYPE_HEADER = "X-Gitlab-Authentication-Type"
        OIDC_AUTH = "oidc"

        def __init__(
            self,
            oidc_auth_provider: AuthProvider,
            bypass_auth: bool,
            path_resolver: _PathResolver,
        ):
            self.oidc_auth_provider = oidc_auth_provider
            self.bypass_auth = bypass_auth
            self.path_resolver = path_resolver

        async def authenticate(
            self, conn: HTTPConnection
        ) -> Optional[Tuple[AuthCredentials, GitLabUser]]:
            """
            Ref: https://www.starlette.io/authentication/
            """

            if self.path_resolver.skip_path(conn.url.path):
                return None

            if self.bypass_auth:
                log.critical("Auth is disabled, all users allowed")

                return AuthCredentials(), GitLabUser(authenticated=True, is_debug=True)

            if self.AUTH_HEADER not in conn.headers:
                raise AuthenticationError("No authorization header presented")

            header = conn.headers[self.AUTH_HEADER]
            bearer, _, token = header.partition(" ")
            if bearer.lower() != self.PREFIX_BEARER_HEADER:
                raise AuthenticationError("Invalid authorization header")

            authenticated, claims = self.authenticate_with_token(conn.headers, token)
            self._validate_headers(claims, conn.headers)

            return AuthCredentials(claims.scopes), GitLabUser(
                authenticated, claims=claims
            )

        @timing("auth_duration_s")
        def authenticate_with_token(self, headers, token) -> Tuple[bool, UserClaims]:
            auth_provider = self._auth_provider(headers)

            user = auth_provider.authenticate(token)
            # We will send this with an HTTP header field going forward since we are
            # retiring direct access to the gateway from clients, which was the main
            # reason this value was carried in the access token.
            if user.claims.gitlab_realm:
                context["gitlab_realm"] = user.claims.gitlab_realm

            if not user.authenticated:
                raise AuthenticationError("Forbidden by auth provider")

            return user.authenticated, user.claims

        def _auth_provider(self, headers):
            auth_type = headers.get(self.AUTH_TYPE_HEADER)

            if auth_type == self.OIDC_AUTH:
                return self.oidc_auth_provider

            raise AuthenticationError(
                "Invalid authentication token type - only OIDC is supported"
            )

        def _validate_headers(self, claims, headers):
            claim_header_mapping = {
                "gitlab_realm": X_GITLAB_REALM_HEADER,
                "subject": X_GITLAB_INSTANCE_ID_HEADER,
            }

            for claim, header in claim_header_mapping.items():
                claim_val = getattr(claims, claim)
                if claim_val and claim_val != headers.get(header):
                    raise AuthenticationError(f"Header mismatch '{header}'")

    @staticmethod
    def on_auth_error(_: Request, e: Exception):
        content = jsonable_encoder({"error": str(e)})
        context["auth_error_details"] = str(e)
        return JSONResponse(status_code=401, content=content)

    def __init__(
        self,
        oidc_auth_provider: AuthProvider,
        bypass_auth: bool = False,
        skip_endpoints: Optional[list] = None,
    ):
        path_resolver = _PathResolver.from_optional_list(skip_endpoints)

        super().__init__(
            AuthenticationMiddleware,
            backend=MiddlewareAuthentication.AuthBackend(
                oidc_auth_provider, bypass_auth, path_resolver
            ),
            on_error=MiddlewareAuthentication.on_auth_error,
        )


class MiddlewareModelTelemetry(Middleware):
    class TelemetryHeadersMiddleware(BaseHTTPMiddleware):
        def __init__(self, path_resolver: _PathResolver, *args, **kwargs):
            self.path_resolver = path_resolver
            self.instrumentator = TelemetryInstrumentator()
            super().__init__(*args, **kwargs)

        async def dispatch(self, request, call_next):
            if self.path_resolver.skip_path(request.url.path):
                return await call_next(request)

            headers = request.headers
            if self._missing_header(headers):
                return await call_next(request)

            try:
                telemetry = Telemetry(
                    accepts=headers.get("X-GitLab-CS-Accepts"),
                    requests=headers.get("X-GitLab-CS-Requests"),
                    errors=headers.get("X-GitLab-CS-Errors"),
                )

                with self.instrumentator.watch([telemetry]):
                    return await call_next(request)
            except ValidationError as e:
                log_exception(e)
                return await call_next(request)

        def _missing_header(self, headers: Headers) -> bool:
            return any(
                value is None
                for value in [
                    headers.get("X-GitLab-CS-Accepts"),
                    headers.get("X-GitLab-CS-Requests"),
                    headers.get("X-GitLab-CS-Errors"),
                ]
            )

    def __init__(self, skip_endpoints: Optional[list] = None):
        path_resolver = _PathResolver.from_optional_list(skip_endpoints)

        super().__init__(
            MiddlewareModelTelemetry.TelemetryHeadersMiddleware,
            path_resolver=path_resolver,
        )
