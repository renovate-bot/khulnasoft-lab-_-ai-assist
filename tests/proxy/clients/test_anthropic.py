import json

import fastapi
import pytest

from ai_gateway.proxy.clients.anthropic import AnthropicProxyClient

from .fixtures import async_client_factory, concurrency_limit, request_factory


@pytest.mark.asyncio
async def test_valid_proxy_request(
    async_client_factory, concurrency_limit, request_factory, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    proxy_client = AnthropicProxyClient(async_client_factory(), concurrency_limit)

    request_params = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
        "stream": True,
    }

    response = await proxy_client.proxy(
        request_factory(
            request_url="http://0.0.0.0:5052/v1/proxy/anthropic/v1/messages",
            request_body=json.dumps(request_params).encode("utf-8"),
            request_headers={
                "accept": "application/json",
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )
    )

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200
