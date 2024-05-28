from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_gateway.api import create_fast_api_server
from ai_gateway.api.monitoring import validated
from ai_gateway.auth import User, UserClaims
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsOutput,
)
from ai_gateway.code_suggestions.processing import ModelEngineOutput
from ai_gateway.code_suggestions.processing.typing import (
    LanguageId,
    MetadataCodeContent,
    MetadataPromptBuilder,
)
from ai_gateway.config import Config, ConfigAuth
from ai_gateway.container import ContainerApplication
from ai_gateway.models import (
    ModelAPIError,
    ModelMetadata,
    SafetyAttributes,
    TextGenModelOutput,
)
from ai_gateway.models.base import TokensConsumptionMetadata


@pytest.fixture
def container() -> ContainerApplication:
    return ContainerApplication()


@pytest.fixture
def fastapi_server_app(container: ContainerApplication) -> Iterator[FastAPI]:
    config = Config(_env_file=None, auth=ConfigAuth())
    container.config.from_dict(config.model_dump())
    yield create_fast_api_server(config)


@pytest.fixture
def client(fastapi_server_app: FastAPI) -> TestClient:
    return TestClient(fastapi_server_app)


# Avoid the global state of checks leaking between tests
@pytest.fixture(autouse=True)
def reset_validated():
    validated.clear()
    yield


def vertex_mock() -> mock.Mock:
    vertex_model_output = [
        ModelEngineOutput(
            text="def search",
            score=0,
            model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
            lang_id=LanguageId.PYTHON,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                },
            ),
            tokens_consumption_metadata=TokensConsumptionMetadata(
                input_tokens=1, output_tokens=2
            ),
        )
    ]
    vertex_completions_mock = mock.Mock(spec=CodeCompletionsLegacy)
    vertex_completions_mock.execute = mock.AsyncMock(return_value=vertex_model_output)

    return vertex_completions_mock


def anthropic_mock() -> mock.Mock:
    anthropic_completions_mock = mock.Mock(spec=CodeCompletions)
    anthropic_model_output = TextGenModelOutput(
        text="def hello_world():",
        score=10_000,
        safety_attributes=SafetyAttributes(),
    )
    anthropic_completions_mock.execute = mock.AsyncMock(
        return_value=anthropic_model_output
    )

    return anthropic_completions_mock


@contextmanager
def engine_mocks(
    container: ContainerApplication,
    vertex: CodeCompletionsLegacy,
    anthropic: CodeCompletions,
):
    with container.code_suggestions.generations.anthropic_factory.override(anthropic):
        with container.code_suggestions.completions.vertex_legacy.override(vertex):
            yield


def test_healthz(client: TestClient):
    response = client.get("/monitoring/healthz")
    assert response.status_code == 200


def test_ready(client: TestClient, container: ContainerApplication):
    anthropic_completions_mock = anthropic_mock()
    vertex_completions_mock = vertex_mock()

    with engine_mocks(
        container=container,
        vertex=vertex_completions_mock,
        anthropic=anthropic_completions_mock,
    ):
        response = client.get("/monitoring/ready")
        response = client.get("/monitoring/ready")

    assert response.status_code == 200
    # assert we only called each model once
    assert vertex_completions_mock.mock_calls == [
        mock.call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]
    assert anthropic_completions_mock.mock_calls == [
        mock.call.execute(
            prefix="def hello_world():",
            file_name="monitoring.py",
            editor_lang="python",
            model_provider="anthropic",
        )
    ]


def model_failure(*args, **kwargs):
    raise ModelAPIError("Vertex unreachable")


def test_ready_vertex_failure(client: TestClient, container: ContainerApplication):
    vertex_completions_mock = vertex_mock()
    vertex_completions_mock.execute = mock.AsyncMock(side_effect=model_failure)

    anthropic_completions_mock = anthropic_mock()
    anthropic_completions_mock.execute = mock.AsyncMock(side_effect=model_failure)

    with engine_mocks(
        container=container,
        vertex=vertex_completions_mock,
        anthropic=anthropic_completions_mock,
    ):
        response = client.get("/monitoring/ready")

    assert vertex_completions_mock.mock_calls == [
        mock.call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]
    # Don't try antrhopic if vertex is not available, no need to spend
    # the money if the service is not going to be ready
    assert not anthropic_completions_mock.mock_calls
    assert response.status_code == 503


def test_ready_anthropic_failure(client: TestClient, container: ContainerApplication):
    vertex_completions_mock = vertex_mock()

    anthropic_completions_mock = anthropic_mock()
    anthropic_completions_mock.execute = mock.AsyncMock(side_effect=model_failure)

    with engine_mocks(
        container=container,
        vertex=vertex_completions_mock,
        anthropic=anthropic_completions_mock,
    ):
        response = client.get("/monitoring/ready")

    assert vertex_completions_mock.mock_calls == [
        mock.call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]
    # Don't try antrhopic if vertex is not available, no need to spend
    # the money if the service is not going to be ready
    assert anthropic_completions_mock.mock_calls == [
        mock.call.execute(
            prefix="def hello_world():",
            file_name="monitoring.py",
            editor_lang="python",
            model_provider="anthropic",
        )
    ]
    assert response.status_code == 503
