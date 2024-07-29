from unittest.mock import patch

import pytest

from ai_gateway.api.v1 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.internal_events import InternalEventAdditionalProperties


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(
            scopes=[
                GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
            ]
        ),
    )


class TestProxyVertexAI:
    def test_successful_request(self, mock_client, mock_track_internal_event):
        with patch(
            "ai_gateway.proxy.clients.VertexAIProxyClient.proxy",
            return_value={"response": "test"},
        ):
            response = mock_client.post(
                "/proxy/vertex-ai",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi, how are you?"}],
                    "stream": "true",
                },
            )

        assert response.status_code == 200
        assert response.json() == {"response": "test"}

        mock_track_internal_event.assert_called_once_with(
            "request_explain_vulnerability",
            category="ai_gateway.api.v1.proxy.vertex_ai",
        )


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_client):
        with patch("ai_gateway.proxy.clients.VertexAIProxyClient.proxy"):
            response = mock_client.post(
                "/proxy/vertex-ai",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hi, how are you?"}],
                    "stream": "true",
                },
            )

        assert response.status_code == 403
        assert response.json() == {
            "detail": "Unauthorized to access explain_vulnerability"
        }
