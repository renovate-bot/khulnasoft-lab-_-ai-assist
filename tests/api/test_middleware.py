from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request
from starlette_context import context, request_cycle_context

from ai_gateway.api.middleware import (
    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER,
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER,
    X_GITLAB_VERSION_HEADER,
    DistributedTraceMiddleware,
    FeatureFlagMiddleware,
    InternalEventMiddleware,
)
from ai_gateway.cloud_connector import X_GITLAB_DUO_SEAT_COUNT_HEADER
from ai_gateway.internal_events import EventContext


@pytest.fixture
def mock_app():
    return AsyncMock()


@pytest.fixture
def internal_event_middleware(mock_app):
    return InternalEventMiddleware(
        mock_app, skip_endpoints=["/health"], enabled=True, environment="test"
    )


@pytest.fixture
def distributed_trace_middleware(mock_app):
    return DistributedTraceMiddleware(
        mock_app, skip_endpoints=["/health"], environment="development"
    )


@pytest.fixture
def feature_flag_middleware(mock_app, disallowed_flags):
    return FeatureFlagMiddleware(mock_app, disallowed_flags)


@pytest.fixture
def disallowed_flags():
    return {}


@pytest.mark.asyncio
async def test_middleware_non_http_request(internal_event_middleware):
    scope = {"type": "websocket"}
    receive = AsyncMock()
    send = AsyncMock()

    with request_cycle_context({}), patch(
        "ai_gateway.api.middleware.current_event_context"
    ) as mock_event_context:
        await internal_event_middleware(scope, receive, send)
        mock_event_context.set.assert_not_called()

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_disabled(mock_app):
    internal_event_middleware = InternalEventMiddleware(
        mock_app, skip_endpoints=[], enabled=False, environment="test"
    )
    request = Request({"type": "http"})
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with request_cycle_context({}), patch(
        "ai_gateway.api.middleware.current_event_context"
    ) as mock_event_context:
        await internal_event_middleware(scope, receive, send)
        mock_event_context.set.assert_not_called()

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_skip_path(internal_event_middleware):
    request = Request(
        {
            "type": "http",
            "path": "/health",
            "headers": [],
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with request_cycle_context({}), patch(
        "ai_gateway.api.middleware.current_event_context"
    ) as mock_event_context:
        await internal_event_middleware(scope, receive, send)
        mock_event_context.set.assert_not_called()

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_set_context(internal_event_middleware):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (b"user-agent", b"TestAgent"),
                (X_GITLAB_REALM_HEADER.lower().encode(), b"test-realm"),
                (X_GITLAB_INSTANCE_ID_HEADER.lower().encode(), b"test-instance"),
                (X_GITLAB_HOST_NAME_HEADER.lower().encode(), b"test-host"),
                (X_GITLAB_VERSION_HEADER.lower().encode(), b"test-version"),
                (X_GITLAB_GLOBAL_USER_ID_HEADER.lower().encode(), b"test-user"),
                (X_GITLAB_DUO_SEAT_COUNT_HEADER.lower().encode(), b"100"),
            ],
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with request_cycle_context({}), patch(
        "ai_gateway.api.middleware.current_event_context"
    ) as mock_event_context, patch(
        "ai_gateway.api.middleware.tracked_internal_events"
    ) as mock_tracked_internal_events:
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm="test-realm",
            instance_id="test-instance",
            host_name="test-host",
            instance_version="test-version",
            global_user_id="test-user",
            duo_seat_count="100",
            feature_enabled_by_namespace_ids=[],
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)
        mock_tracked_internal_events.set.assert_called_once_with(set())
        assert context["tracked_internal_events"] == []

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers, expected",
    [
        (
            [
                (
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"1,2,3",
                )
            ],
            [1, 2, 3],
        ),
        (
            [
                (
                    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"4,5,6",
                )
            ],
            [4, 5, 6],
        ),
        (
            [
                (
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"",
                )
            ],
            [],
        ),
        (
            [
                (
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER.lower().encode(),
                    b"1,2,a",
                )
            ],
            None,
        ),
    ],
)
async def test_middleware_set_context_feature_enabled_by_namespace_ids(
    internal_event_middleware, headers, expected
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with request_cycle_context({}), patch(
        "ai_gateway.api.middleware.current_event_context"
    ) as mock_event_context:
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm=None,
            instance_id=None,
            host_name=None,
            instance_version=None,
            global_user_id=None,
            feature_enabled_by_namespace_ids=expected,
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_missing_headers(internal_event_middleware):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (b"user-agent", b"TestAgent"),
            ],
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with request_cycle_context({}), patch(
        "ai_gateway.api.middleware.current_event_context"
    ) as mock_event_context:
        await internal_event_middleware(scope, receive, send)

        expected_context = EventContext(
            environment="test",
            source="ai-gateway-python",
            realm=None,
            instance_id=None,
            host_name=None,
            instance_version=None,
            global_user_id=None,
            duo_seat_count=None,
            feature_enabled_by_namespace_ids=[],
            context_generated_at=mock_event_context.set.call_args[0][
                0
            ].context_generated_at,
        )
        mock_event_context.set.assert_called_once_with(expected_context)

    internal_event_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
async def test_middleware_distributed_trace(distributed_trace_middleware):
    current_run_id = "20240808T090953171943Z18dfa1db-1dfc-4a48-aaf8-a139960955ce"
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": [
                (b"langsmith-trace", current_run_id.encode()),
            ],
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with patch("ai_gateway.api.middleware.tracing_context") as mock_tracing_context:
        await distributed_trace_middleware(scope, receive, send)

        mock_tracing_context.assert_called_once_with(parent=current_run_id)

    distributed_trace_middleware.app.assert_called_once_with(scope, receive, send)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,disallowed_flags,expected_flags",
    [
        (
            [
                (b"x-gitlab-enabled-feature-flags", b"feature_a,feature_b,feature_c"),
            ],
            {},
            {"feature_a", "feature_b", "feature_c"},
        ),
        (
            [
                (b"x-gitlab-enabled-feature-flags", b"feature_a,feature_b,feature_c"),
                (b"x-gitlab-realm", b"self-managed"),
            ],
            {"self-managed": {"feature_a"}},
            {"feature_b", "feature_c"},
        ),
    ],
)
async def test_middleware_feature_flag(
    feature_flag_middleware, headers, disallowed_flags, expected_flags
):
    request = Request(
        {
            "type": "http",
            "path": "/api/endpoint",
            "headers": headers,
        }
    )
    scope = request.scope
    receive = AsyncMock()
    send = AsyncMock()

    with patch(
        "ai_gateway.api.middleware.current_feature_flag_context"
    ) as mock_feature_flag_context, request_cycle_context({}):
        await feature_flag_middleware(scope, receive, send)

        mock_feature_flag_context.set.assert_called_once_with(expected_flags)

        assert set(context["enabled_feature_flags"].split(",")) == expected_flags

    feature_flag_middleware.app.assert_called_once_with(scope, receive, send)
