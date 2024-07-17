import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from ai_gateway.models.container import (
    _init_anthropic_proxy_client,
    _init_vertex_ai_proxy_client,
    _init_vertex_grpc_client,
)


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
