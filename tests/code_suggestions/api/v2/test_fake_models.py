import pytest
from fastapi.testclient import TestClient

from ai_gateway.api.v2.api import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.code_suggestions import CodeCompletionsLegacy, CodeGenerations
from ai_gateway.code_suggestions.processing import ModelEngineCompletions
from ai_gateway.code_suggestions.processing.post.completions import PostProcessor
from ai_gateway.code_suggestions.processing.pre import TokenizerTokenStrategy
from ai_gateway.deps import CodeSuggestionsContainer
from ai_gateway.experimentation import ExperimentRegistry
from ai_gateway.models import FakePalmTextGenModel
from ai_gateway.tokenizer import init_tokenizer


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(scopes=["code_suggestions"]),
    )


class TestFakeModels:
    # Verify fake models with most used routes

    def test_fake_completions(self, mock_client: TestClient):
        """Completions: v1 with Vertex AI models."""

        container = CodeSuggestionsContainer()
        engine = ModelEngineCompletions(
            model=FakePalmTextGenModel(),
            tokenizer=init_tokenizer(),
            experiment_registry=ExperimentRegistry(),
        )

        code_completions_mock = CodeCompletionsLegacy(
            engine=engine,
            post_processor=PostProcessor,
        )

        container.code_completions_legacy.override(code_completions_mock)

        response = mock_client.post(
            "/v2/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
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
            },
        )

        assert response.status_code == 200

        body = response.json()
        assert body["choices"][0]["text"] == "fake code suggestion from PaLM Text"

    def test_fake_generations(self, mock_client: TestClient):
        """Generations: v2 with Anthropic models."""

        container = CodeSuggestionsContainer()
        tokenization_strategy = TokenizerTokenStrategy(init_tokenizer())

        code_generations_mock = CodeGenerations(
            model=FakePalmTextGenModel(), tokenization_strategy=tokenization_strategy
        )

        container.code_generations_anthropic.override(code_generations_mock)

        response = mock_client.post(
            "/v2/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
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
        assert body["choices"][0]["text"] == "fake code suggestion from PaLM Text"
