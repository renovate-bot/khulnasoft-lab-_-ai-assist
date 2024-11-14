import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Optional, Tuple

import structlog
from asgi_correlation_id.context import correlation_id
from fastapi.encoders import jsonable_encoder
from gitlab_cloud_connector import (
    X_GITLAB_DUO_SEAT_COUNT_HEADER,
    AuthProvider,
    CloudConnectorAuthError,
    CloudConnectorUser,
)
from gitlab_cloud_connector import authenticate as cloud_connector_authenticate
from langsmith.run_helpers import tracing_context
from pydantic import ValidationError
from starlette.authentication import (
    AuthCredentials,
    AuthenticationError,
    HTTPConnection,
)
from starlette.datastructures import CommaSeparatedStrings, Headers, MutableHeaders
from starlette.middleware import Middleware
from starlette.middleware.authentication import (
    AuthenticationBackend,
    AuthenticationMiddleware,
)
from starlette.middleware.base import BaseHTTPMiddleware, Request
from starlette.responses import JSONResponse
from starlette_context import context as starlette_context
from uvicorn.protocols.utils import get_path_with_query_string

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.api.timing import timing
from ai_gateway.feature_flags import current_feature_flag_context
from ai_gateway.instrumentators.base import Telemetry, TelemetryInstrumentator
from ai_gateway.internal_events import (
    EventContext,
    current_event_context,
    tracked_internal_events,
)
from ai_gateway.tracking.errors import log_exception

__all__ = [
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
X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER = "X-Gitlab-Saas-Duo-Pro-Namespace-Ids"
X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER = (
    "X-Gitlab-Feature-Enabled-By-Namespace-Ids"
)
X_GITLAB_MODEL_GATEWAY_REQUEST_SENT_AT = "X-Gitlab-Rails-Send-Start"
X_GITLAB_LANGUAGE_SERVER_VERSION = "X-Gitlab-Language-Server-Version"
X_GITLAB_ENABLED_FEATURE_FLAGS = "x-gitlab-enabled-feature-flags"


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


class AccessLogMiddleware:
    """Middleware for access logging."""

    def __init__(self, app, skip_endpoints):
        self.app = app
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self.path_resolver.skip_path(request.url.path):
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        # These context vars will be added to all log entries emitted during the request
        request_id = correlation_id.get()
        structlog.contextvars.bind_contextvars(correlation_id=request_id)

        start_time_total = time.perf_counter()
        start_time_cpu = time.process_time()
        response_start_duration_s = 0.0
        first_chunk_duration_s = 0.0
        request_arrived_at = datetime.now(timezone.utc)
        # duration_request represents latency added by sending request from Rails to AI gateway
        try:
            wait_duration = time.time() - float(
                request.headers.get(X_GITLAB_MODEL_GATEWAY_REQUEST_SENT_AT)
            )
        except (ValueError, TypeError):
            wait_duration = -1

        status_code = 500
        content_type = "unknown"

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                nonlocal status_code, start_time_total, response_start_duration_s, first_chunk_duration_s, content_type
                status_code = message["status"]

                headers = MutableHeaders(scope=message)

                if "content-type" in headers:
                    content_type = headers["content-type"]

                response_start_duration_s = time.perf_counter() - start_time_total
                headers.append("X-Process-Time", str(response_start_duration_s))

            if message["type"] == "http.response.body":
                if first_chunk_duration_s == 0.0:
                    first_chunk_duration_s = time.perf_counter() - start_time_total

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            starlette_context.data["exception_message"] = str(e)
            starlette_context.data["exception_class"] = type(e).__name__
            starlette_context.data["exception_backtrace"] = traceback.format_exc()
            log_exception(e)
            raise e
        finally:
            elapsed_time = time.perf_counter() - start_time_total
            cpu_time = time.process_time() - start_time_cpu
            url = get_path_with_query_string(request.scope)
            client_host = request.client.host
            client_port = request.client.port
            http_method = request.method
            http_version = request.scope["http_version"]

            fields = {
                "url": str(request.url),
                "path": url,
                "status_code": status_code,
                "method": http_method,
                "correlation_id": request_id,
                "http_version": http_version,
                "client_ip": client_host,
                "client_port": client_port,
                "duration_s": elapsed_time,
                "duration_request": wait_duration,
                "request_arrived_at": request_arrived_at.isoformat(),
                "response_start_duration_s": response_start_duration_s,
                "first_chunk_duration_s": first_chunk_duration_s,
                "cpu_s": cpu_time,
                "content_type": content_type,
                "user_agent": request.headers.get("User-Agent"),
                "gitlab_language_server_version": request.headers.get(
                    X_GITLAB_LANGUAGE_SERVER_VERSION
                ),
                "gitlab_instance_id": request.headers.get(X_GITLAB_INSTANCE_ID_HEADER),
                "gitlab_global_user_id": request.headers.get(
                    X_GITLAB_GLOBAL_USER_ID_HEADER
                ),
                "gitlab_host_name": request.headers.get(X_GITLAB_HOST_NAME_HEADER),
                "gitlab_version": request.headers.get(X_GITLAB_VERSION_HEADER),
                "gitlab_saas_duo_pro_namespace_ids": request.headers.get(
                    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER
                ),
                "gitlab_feature_enabled_by_namespace_ids": request.headers.get(
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER
                ),
                "gitlab_realm": request.headers.get(X_GITLAB_REALM_HEADER),
                "gitlab_duo_seat_count": request.headers.get(
                    X_GITLAB_DUO_SEAT_COUNT_HEADER
                ),
            }
            fields.update(starlette_context.data)

            # Recreate the Uvicorn access log format, but add all parameters as structured information
            access_logger.info(
                f"""{client_host}:{client_port} - "{http_method} {url} HTTP/{http_version}" {status_code}""",
                **fields,
            )


class MiddlewareAuthentication(Middleware):
    class AuthBackend(AuthenticationBackend):

        def __init__(
            self,
            oidc_auth_provider: AuthProvider,
            bypass_auth: bool,
            bypass_auth_with_header: bool,
            path_resolver: _PathResolver,
        ):
            self.oidc_auth_provider = oidc_auth_provider
            self.bypass_auth = bypass_auth
            self.bypass_auth_with_header = bypass_auth_with_header
            self.path_resolver = path_resolver

        async def authenticate(
            self, conn: HTTPConnection
        ) -> Optional[Tuple[AuthCredentials, StarletteUser]]:
            """
            Ref: https://www.starlette.io/authentication/
            """

            if self.path_resolver.skip_path(conn.url.path):
                return None

            if self.bypass_auth:
                log.critical("Auth is disabled, all users allowed")
                cloud_connector_user, _cloud_connector_error = (
                    cloud_connector_authenticate(
                        dict(conn.headers), None, bypass_auth=True
                    )
                )

                return AuthCredentials(), StarletteUser(cloud_connector_user)

            if (
                self.bypass_auth_with_header  # Should only be set and used for test & dev
                and conn.headers.get("Bypass-Auth") == "true"
            ):
                log.critical(
                    "Auth is disabled, all requests with `Bypass-Auth` header set allowed"
                )
                cloud_connector_user, _cloud_connector_error = (
                    cloud_connector_authenticate(
                        dict(conn.headers), None, bypass_auth=True
                    )
                )

                return AuthCredentials(), StarletteUser(cloud_connector_user)

            cloud_connector_user, cloud_connector_error = self.cloud_connector_auth(
                conn.headers
            )

            if hasattr(cloud_connector_user.claims, "issuer"):
                starlette_context["token_issuer"] = cloud_connector_user.claims.issuer

            # We will send this with an HTTP header field going forward since we are
            # retiring direct access to the gateway from clients, which was the main
            # reason this value was carried in the access token.
            if (
                hasattr(cloud_connector_user.claims, "gitlab_realm")
                and cloud_connector_user.claims.gitlab_realm
            ):
                starlette_context["gitlab_realm"] = (
                    cloud_connector_user.claims.gitlab_realm
                )

            if cloud_connector_error:
                raise AuthenticationError(cloud_connector_error.error_message)

            return AuthCredentials(cloud_connector_user.claims.scopes), StarletteUser(
                cloud_connector_user
            )

        @timing("auth_duration_s")
        def cloud_connector_auth(
            self, headers
        ) -> Tuple[CloudConnectorUser, Optional[CloudConnectorAuthError]]:
            return cloud_connector_authenticate(dict(headers), self.oidc_auth_provider)

    @staticmethod
    def on_auth_error(_: Request, e: Exception):
        content = jsonable_encoder({"error": str(e)})
        starlette_context["auth_error_details"] = str(e)
        starlette_context["http_exception_details"] = str(e)
        return JSONResponse(status_code=401, content=content)

    def __init__(
        self,
        oidc_auth_provider: AuthProvider,
        bypass_auth: bool = False,
        bypass_auth_with_header: bool = False,
        skip_endpoints: Optional[list] = None,
    ):
        path_resolver = _PathResolver.from_optional_list(skip_endpoints)

        super().__init__(
            AuthenticationMiddleware,
            backend=MiddlewareAuthentication.AuthBackend(
                oidc_auth_provider,
                bypass_auth,
                bypass_auth_with_header,
                path_resolver,
            ),
            on_error=MiddlewareAuthentication.on_auth_error,
        )


class InternalEventMiddleware:
    def __init__(self, app, skip_endpoints, enabled, environment):
        self.app = app
        self.enabled = enabled
        self.environment = environment
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self.path_resolver.skip_path(request.url.path):
            await self.app(scope, receive, send)
            return

        # Fetching a list of namespaces that allow the user to use the tracked feature.
        # This is relevant for requests coming from gitlab.com, and unrelated to self-managed or dedicated instances.
        feature_enabled_by_namespace_ids = list(
            CommaSeparatedStrings(
                request.headers.get(
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER, ""
                )
            )
        )
        # Supporting the legacy header https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/561.
        if not feature_enabled_by_namespace_ids:
            feature_enabled_by_namespace_ids = list(
                CommaSeparatedStrings(
                    request.headers.get(X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER, "")
                )
            )

        try:
            feature_enabled_by_namespace_ids = [
                int(str_id) for str_id in feature_enabled_by_namespace_ids
            ]
        except ValueError:
            feature_enabled_by_namespace_ids = None

        context = EventContext(
            environment=self.environment,
            source="ai-gateway-python",
            realm=request.headers.get(X_GITLAB_REALM_HEADER),
            instance_id=request.headers.get(X_GITLAB_INSTANCE_ID_HEADER),
            host_name=request.headers.get(X_GITLAB_HOST_NAME_HEADER),
            instance_version=request.headers.get(X_GITLAB_VERSION_HEADER),
            global_user_id=request.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER),
            feature_enabled_by_namespace_ids=feature_enabled_by_namespace_ids,
            context_generated_at=datetime.now().isoformat(),
            correlation_id=correlation_id.get(),
        )
        current_event_context.set(context)
        tracked_internal_events.set(set())

        await self.app(scope, receive, send)

        starlette_context["tracked_internal_events"] = list(
            tracked_internal_events.get()
        )


class DistributedTraceMiddleware:
    """Middleware for distributed tracing."""

    def __init__(self, app, skip_endpoints, environment):
        self.app = app
        self.environment = environment
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self.path_resolver.skip_path(request.url.path):
            await self.app(scope, receive, send)
            return

        if self.environment == "development" and "langsmith-trace" in request.headers:
            # Set the distrubted tracing LangSmith header to the tracing context, which is sent from Langsmith::RunHelpers of GitLab-Rails/Sidekiq.
            # See https://docs.gitlab.com/ee/development/ai_features/duo_chat.html#tracing-with-langsmith
            # and https://docs.smith.langchain.com/how_to_guides/tracing/distributed_tracing
            with tracing_context(parent=request.headers["langsmith-trace"]):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


class FeatureFlagMiddleware:
    """Middleware for feature flags."""

    def __init__(self, app, disallowed_flags: dict = None):
        self.app = app
        self.disallowed_flags = disallowed_flags

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if X_GITLAB_ENABLED_FEATURE_FLAGS not in request.headers:
            await self.app(scope, receive, send)
            return

        enabled_feature_flags = request.headers.get(
            X_GITLAB_ENABLED_FEATURE_FLAGS, ""
        ).split(",")
        enabled_feature_flags = set(enabled_feature_flags)

        if self.disallowed_flags:
            # Remove feature flags that are not supported in the specific realm.
            gitlab_realm = request.headers.get(X_GITLAB_REALM_HEADER, "")
            disallowed_flags = self.disallowed_flags.get(gitlab_realm, set())
            enabled_feature_flags = enabled_feature_flags.difference(disallowed_flags)

        current_feature_flag_context.set(enabled_feature_flags)
        starlette_context["enabled_feature_flags"] = ",".join(enabled_feature_flags)

        await self.app(scope, receive, send)


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
