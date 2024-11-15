import pytest
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.api.v2 import api_router
from ai_gateway.config import Config


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["complete_code", "generate_code"],
            subject="1234",
            gitlab_realm="self-managed",
        ),
    )


@pytest.fixture
def mock_config():
    yield Config(mock_model_responses=True)


class TestMockedModels:
    # Verify mocked models with most used routes

    def test_completions(
        self,
        mock_client: TestClient,
    ):
        """Completions: v1 with Vertex AI models."""
        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_version": 1,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "def beautiful_",
                    "content_below_cursor": "\n",
                },
                "choices_count": 1,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["choices"][0]["text"].startswith("echo:")

    def test_fake_generations(self, mock_client: TestClient):
        """Generations: v2 with Anthropic models."""
        response = mock_client.post(
            "/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_version": 2,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "wonder",
                    "content_below_cursor": "\n",
                },
                "prompt": "write a wonderful function",
                "model_provider": "anthropic",
            },
        )

        assert response.status_code == 200

        body = response.json()
        assert body["choices"][0]["text"].startswith("echo:")
