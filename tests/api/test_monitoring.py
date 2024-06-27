from typing import Iterator
from unittest.mock import Mock, call

import pytest
from dependency_injector import containers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_gateway.api import create_fast_api_server
from ai_gateway.api.monitoring import validated
from ai_gateway.config import Config, ConfigAuth
from ai_gateway.models import ModelAPIError


@pytest.fixture
def mock_config():
    yield Config(_env_file=None, auth=ConfigAuth())


@pytest.fixture
def fastapi_server_app(mock_config: Config) -> Iterator[FastAPI]:
    yield create_fast_api_server(mock_config)


@pytest.fixture
def client(
    fastapi_server_app: FastAPI, mock_container: containers.Container
) -> TestClient:
    return TestClient(fastapi_server_app)


# Avoid the global state of checks leaking between tests
@pytest.fixture(autouse=True)
def reset_validated():
    validated.clear()
    yield


def test_healthz(client: TestClient):
    response = client.get("/monitoring/healthz")
    assert response.status_code == 200


def test_ready(
    client: TestClient, mock_generations: Mock, mock_completions_legacy: Mock
):
    response = client.get("/monitoring/ready")
    response = client.get("/monitoring/ready")

    assert response.status_code == 200
    # assert we only called each model once
    assert mock_completions_legacy.mock_calls == [
        call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]
    assert mock_generations.mock_calls == [
        call.execute(
            prefix="\n\nHuman: Complete this code: def hello_world():\n\nAssistant:",
            file_name="monitoring.py",
            editor_lang="python",
            model_provider="anthropic",
        )
    ]


def model_failure(*args, **kwargs):
    raise ModelAPIError("Vertex unreachable")


def test_ready_vertex_failure(
    client: TestClient, mock_generations: Mock, mock_completions_legacy: Mock
):
    mock_generations.side_effect = model_failure
    mock_completions_legacy.side_effect = model_failure

    response = client.get("/monitoring/ready")

    assert mock_completions_legacy.mock_calls == [
        call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]
    # Don't try antrhopic if vertex is not available, no need to spend
    # the money if the service is not going to be ready
    assert not mock_generations.mock_calls
    assert response.status_code == 503


def test_ready_anthropic_failure(
    client: TestClient, mock_generations: Mock, mock_completions_legacy: Mock
):
    mock_generations.side_effect = model_failure

    response = client.get("/monitoring/ready")

    assert mock_completions_legacy.mock_calls == [
        call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]
    # Don't try antrhopic if vertex is not available, no need to spend
    # the money if the service is not going to be ready
    assert mock_generations.mock_calls == [
        call.execute(
            prefix="\n\nHuman: Complete this code: def hello_world():\n\nAssistant:",
            file_name="monitoring.py",
            editor_lang="python",
            model_provider="anthropic",
        )
    ]
    assert response.status_code == 503
