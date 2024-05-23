from unittest import mock

import pytest
from dependency_injector import containers
from fastapi.testclient import TestClient

from ai_gateway.api.v2 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.code_suggestions import CodeCompletionsLegacy, CodeGenerations
from ai_gateway.code_suggestions.processing import ModelEngineCompletions
from ai_gateway.code_suggestions.processing.base import ModelEngineOutput
from ai_gateway.code_suggestions.processing.post.completions import PostProcessor
from ai_gateway.code_suggestions.processing.pre import TokenizerTokenStrategy
from ai_gateway.code_suggestions.processing.typing import (
    LanguageId,
    MetadataCodeContent,
    MetadataPromptBuilder,
)
from ai_gateway.experimentation import ExperimentRegistry
from ai_gateway.experimentation.base import ExperimentTelemetry
from ai_gateway.models import (
    AnthropicModel,
    KindAnthropicModel,
    KindVertexTextModel,
    ModelMetadata,
    PalmCodeGeckoModel,
)
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.models.mock import LLM, ChatModel
from ai_gateway.tokenizer import init_tokenizer
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(
            scopes=["code_suggestions"], subject="1234", gitlab_realm="self-managed"
        ),
    )


class TestMockedModels:
    # Verify mocked models with most used routes

    def test_completions(
        self,
        mock_client: TestClient,
        mock_container: containers.DeclarativeContainer,
    ):
        """Completions: v1 with Vertex AI models."""

        engine = ModelEngineCompletions(
            model=LLM(),
            tokenization_strategy=TokenizerTokenStrategy(init_tokenizer()),
            experiment_registry=ExperimentRegistry(),
        )

        model_output = [
            ModelEngineOutput(
                text="echo: ",
                score=0,
                model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                lang_id=LanguageId.PYTHON,
                metadata=MetadataPromptBuilder(
                    components={
                        "prefix": MetadataCodeContent(length=10, length_tokens=2),
                        "suffix": MetadataCodeContent(length=10, length_tokens=2),
                    },
                    experiments=[
                        ExperimentTelemetry(name="truncate_suffix", variant=1)
                    ],
                ),
                tokens_consumption_metadata=TokensConsumptionMetadata(
                    input_tokens=0, output_tokens=0
                ),
            )
        ]

        code_completions_mock = CodeCompletionsLegacy(
            engine=engine,
            post_processor=PostProcessor,
            snowplow_instrumentator=mock.Mock(spec=SnowplowInstrumentator),
        )
        code_completions_mock.execute = mock.AsyncMock(return_value=model_output)

        with mock_container.code_suggestions.completions.vertex_legacy.override(
            code_completions_mock
        ):
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

    def test_fake_generations(
        self, mock_client: TestClient, mock_container: containers.DeclarativeContainer
    ):
        """Generations: v2 with Anthropic models."""

        tokenization_strategy = TokenizerTokenStrategy(init_tokenizer())

        code_generations_mock = CodeGenerations(
            model=LLM(),
            tokenization_strategy=tokenization_strategy,
            snowplow_instrumentator=mock.Mock(spec=SnowplowInstrumentator),
        )

        with mock_container.code_suggestions.generations.anthropic_factory.override(
            code_generations_mock
        ):
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
