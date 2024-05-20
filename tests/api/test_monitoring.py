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
    CodeGenerations,
    CodeSuggestionsOutput,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId
from ai_gateway.config import Config, ConfigAuth
from ai_gateway.container import ContainerApplication
from ai_gateway.models import (
    ModelAPIError,
    ModelMetadata,
    SafetyAttributes,
    TextGenModelOutput,
)


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


def test_healthz(client: TestClient):
    response = client.get("/monitoring/healthz")
    assert response.status_code == 200


def test_ready(client: TestClient, container: ContainerApplication):
    vertex_model_output = CodeSuggestionsOutput(
        text="def search",
        score=0,
        model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
        lang_id=LanguageId.PYTHON,
        metadata=CodeSuggestionsOutput.Metadata(
            experiments=[],
        ),
    )
    vertex_completions_mock = mock.Mock(spec=CodeGenerations)
    vertex_completions_mock.execute = mock.AsyncMock(return_value=vertex_model_output)

    anthropic_completions_mock = mock.Mock(spec=CodeCompletions)
    anthropic_model_output = TextGenModelOutput(
        text="def hello_world():",
        score=10_000,
        safety_attributes=SafetyAttributes(),
    )
    anthropic_completions_mock.execute = mock.AsyncMock(
        return_value=anthropic_model_output
    )

    with container.code_suggestions.generations.anthropic_factory.override(
        anthropic_completions_mock
    ):
        with container.code_suggestions.generations.vertex.override(
            vertex_completions_mock
        ):
            response = client.get("/monitoring/ready")
            response = client.get("/monitoring/ready")

    assert response.status_code == 200
    # assert we only called each model once
    assert vertex_completions_mock.mock_calls == [
        mock.call.execute(
            prefix="def hello_world():",
            file_name="monitoring.py",
            editor_lang="python",
            model_provider="vertex-ai",
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


def test_ready_failure(client: TestClient, container: ContainerApplication):
    def model_failure(*args, **kwargs):
        raise ModelAPIError("Vertex unreachable")

    vertex_completions_mock = mock.Mock(spec=CodeGenerations)
    vertex_completions_mock.execute = mock.AsyncMock(side_effect=model_failure)

    anthropic_completions_mock = mock.Mock(spec=CodeCompletions)
    anthropic_completions_mock.execute = mock.AsyncMock(side_effect=model_failure)

    with container.code_suggestions.generations.anthropic_factory.override(
        anthropic_completions_mock
    ):
        with container.code_suggestions.generations.vertex.override(
            vertex_completions_mock
        ):
            response = client.get("/monitoring/ready")

    assert vertex_completions_mock.mock_calls == [
        mock.call.execute(
            prefix="def hello_world():",
            file_name="monitoring.py",
            editor_lang="python",
            model_provider="vertex-ai",
        )
    ]
    # Don't try antrhopic if vertex is not available, no need to spend
    # the money if the service is not going to be ready
    assert not anthropic_completions_mock.mock_calls
    assert response.status_code == 503
