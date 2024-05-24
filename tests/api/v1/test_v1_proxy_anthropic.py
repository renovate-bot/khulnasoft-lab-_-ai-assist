from unittest import mock

import pytest

from ai_gateway.api.v1 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.container import ContainerApplication
from ai_gateway.gitlab_features import GitLabUnitPrimitive


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


class TestProxyAnthropic:
    def test_successful_request(
        self,
        mock_client,
    ):
        mock_proxy_client = mock.Mock()
        mock_proxy_client.proxy = mock.AsyncMock(return_value={"response": "test"})
        container = ContainerApplication()

        with container.pkg_models.anthropic_proxy_client.override(mock_proxy_client):
            response = mock_client.post(
                "/proxy/anthropic",
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


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_client):
        container = ContainerApplication()

        with container.pkg_models.anthropic_proxy_client.override(mock.Mock()):
            response = mock_client.post(
                "/proxy/anthropic",
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
