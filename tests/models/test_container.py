from typing import cast
from unittest.mock import patch

import httpx
import pytest
from dependency_injector import containers, providers

from ai_gateway.models.container import (
    _init_anthropic_proxy_client,
    _init_vertex_ai_proxy_client,
    _init_vertex_grpc_client,
)
from ai_gateway.proxy.clients.anthropic import AnthropicProxyClient
from ai_gateway.proxy.clients.vertex_ai import VertexAIProxyClient


@pytest.mark.parametrize(
    ("args", "expected_init"),
    [
        (
            {
                "endpoint": "test",
                "mock_model_responses": False,
                "custom_models_enabled": False,
            },
            True,
        ),
        (
            {
                "endpoint": "test",
                "mock_model_responses": False,
                "custom_models_enabled": True,
            },
            False,
        ),
        (
            {
                "endpoint": "test",
                "mock_model_responses": True,
                "custom_models_enabled": False,
            },
            False,
        ),
    ],
)
def test_init_vertex_grpc_client(args, expected_init):
    with patch(
        # "google.cloud.aiplatform.gapic.PredictionServiceAsyncClient"
        "ai_gateway.models.container.grpc_connect_vertex"
    ) as mock_grpc_client:
        _init_vertex_grpc_client(**args)

        if expected_init:
            mock_grpc_client.assert_called_once_with({"api_endpoint": args["endpoint"]})
        else:
            mock_grpc_client.assert_not_called()


@pytest.mark.parametrize(
    ("args", "expected_init"),
    [
        (
            {"mock_model_responses": False},
            True,
        ),
        (
            {"mock_model_responses": True},
            False,
        ),
    ],
)
def test_anthropic_proxy_client(args, expected_init):
    with patch("httpx.AsyncClient") as mock_httpx_client:
        _init_anthropic_proxy_client(**args)

        if expected_init:
            mock_httpx_client.assert_called_once_with(
                base_url="https://api.anthropic.com/",
                timeout=httpx.Timeout(timeout=60.0),
            )
        else:
            mock_httpx_client.assert_not_called()


@pytest.mark.parametrize(
    ("args", "expected_init"),
    [
        (
            {
                "model_keys": {"fireworks_api_key": "test_fireworks_key"},
                "model_endpoints": {
                    "fireworks_current_region_endpoint": {
                        "endpoint": "https://test.fireworks.ai/"
                    }
                },
            },
            True,
        ),
        ({}, False),
    ],
)
def _init_async_fireworks_client(args, expected_init):
    with patch("AsyncOpenAI") as mock_openai_client:
        _init_async_fireworks_client(**args)

        if expected_init:
            mock_openai_client.assert_called_once_with(
                api_key="test_fireworks_key", base_url="https://test.fireworks.ai/"
            )
        else:
            mock_openai_client.assert_not_called()


@pytest.mark.parametrize(
    ("args", "expected_init"),
    [
        (
            {
                "mock_model_responses": False,
                "endpoint": "us-central1-aiplatform.googleapis.com",
            },
            True,
        ),
        (
            {
                "mock_model_responses": True,
                "endpoint": "us-central1-aiplatform.googleapis.com",
            },
            False,
        ),
    ],
)
def test_vertex_ai_proxy_client(args, expected_init):
    with patch("httpx.AsyncClient") as mock_httpx_client:
        _init_vertex_ai_proxy_client(**args)

        if expected_init:
            mock_httpx_client.assert_called_once_with(
                base_url="https://us-central1-aiplatform.googleapis.com/",
                timeout=httpx.Timeout(timeout=60.0),
            )
        else:
            mock_httpx_client.assert_not_called()


@pytest.mark.asyncio
async def test_container(mock_container: containers.DeclarativeContainer):
    models = cast(providers.Container, mock_container.pkg_models)

    assert isinstance(models.anthropic_proxy_client(), AnthropicProxyClient)
    assert isinstance(models.vertex_ai_proxy_client(), VertexAIProxyClient)
