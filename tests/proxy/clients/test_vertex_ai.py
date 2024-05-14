import json
from unittest.mock import ANY, patch

import fastapi
import pytest
from starlette.datastructures import URL

from ai_gateway.proxy.clients.vertex_ai import VertexAIProxyClient

from .fixtures import async_client_factory, concurrency_limit, request_factory


@pytest.mark.asyncio
async def test_valid_proxy_request(
    async_client_factory, concurrency_limit, request_factory
):
    proxy_client = VertexAIProxyClient(
        project="",
        location="",
        client=async_client_factory(),
        concurrency_limit=concurrency_limit,
    )

    request_params = {
        "instances": [{"prefix": "print", "suffix": ""}],
        "parameters": {"temperature": 0.2, "maxOutputTokens": 64},
    }

    with patch("ai_gateway.proxy.clients.vertex_ai.access_token") as mock_access_token:
        response = await proxy_client.proxy(
            request_factory(
                request_url="http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/code-gecko:predict",
                request_body=json.dumps(request_params).encode("utf-8"),
                request_headers={
                    "content-type": "application/json",
                },
            )
        )

        mock_access_token.assert_called_once()

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_url", "expected_upstream_path", "expected_error"),
    [
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/code-gecko:predict",
            "/v1/projects/my-project/locations/my-location/publishers/google/models/code-gecko:predict",
            None,
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/code-gecko:predict?alt=sse",
            "/v1/projects/my-project/locations/my-location/publishers/google/models/code-gecko:predict?alt=sse",
            None,
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/code-gecko:predict?alt=unknown",
            "/v1/projects/my-project/locations/my-location/publishers/google/models/code-gecko:predict",
            None,
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/unknown:predict",
            "",
            "400: Unsupported model",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/code-gecko:unknown",
            "",
            "404: Not found",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/unknown/PROJECT/locations/LOCATION/publishers/google/models/code-gecko:predict",
            "",
            "404: Not found",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/v1/projects/PROJECT/unknown/LOCATION/publishers/google/models/code-gecko:predict",
            "",
            "404: Not found",
        ),
        (
            "http://0.0.0.0:5052/v1/proxy/vertex-ai/corrupted-path/code-gecko:predict",
            "",
            "404: Not found",
        ),
    ],
)
async def test_request_url(
    async_client_factory,
    concurrency_limit,
    request_factory,
    request_url,
    expected_upstream_path,
    expected_error,
):
    async_client = async_client_factory()
    proxy_client = VertexAIProxyClient(
        project="my-project",
        location="my-location",
        client=async_client,
        concurrency_limit=concurrency_limit,
    )

    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            await proxy_client.proxy(request_factory(request_url=request_url))
    else:
        with patch(
            "ai_gateway.proxy.clients.vertex_ai.access_token"
        ) as mock_access_token:
            response = await proxy_client.proxy(
                request_factory(request_url=request_url)
            )

            mock_access_token.assert_called_once()

        assert response.status_code == 200

        async_client.build_request.assert_called_once_with(
            "POST",
            URL(expected_upstream_path),
            headers=ANY,
            json=ANY,
        )
