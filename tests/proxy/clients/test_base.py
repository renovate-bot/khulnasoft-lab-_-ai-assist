from unittest.mock import ANY

import fastapi
import pytest
from starlette.datastructures import URL

from ai_gateway.proxy.clients.base import BaseProxyClient

from .fixtures import async_client_factory, concurrency_limit, request_factory


class TestProxyClient(BaseProxyClient):
    __test__ = False

    def _allowed_upstream_paths(self):
        return ["/valid_path"]

    def _allowed_headers_to_upstream(self):
        return ["Content-Type"]

    def _allowed_headers_to_downstream(self):
        return ["Content-Length"]

    def _upstream_service(self):
        return "test_service"

    def _allowed_upstream_models(self):
        return ["model1", "model2"]

    def _extract_model_name(self, upstream_path, json_body):
        return json_body.get("model")

    def _extract_stream_flag(self, upstream_path, json_body):
        return json_body.get("stream", False)

    def _update_headers_to_upstream(self, headers):
        headers.update({"X-Test-Header": "test"})


@pytest.mark.asyncio
async def test_valid_proxy_request(
    async_client_factory, concurrency_limit, request_factory
):
    proxy_client = TestProxyClient(async_client_factory(), concurrency_limit)

    response = await proxy_client.proxy(request_factory())

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_url", "expected_error"),
    [
        ("http://0.0.0.0:5052/v1/proxy/test_service/valid_path", None),
        ("http://0.0.0.0:5052/v1/proxy/test_service/invalid_path", "404: Not found"),
        ("http://0.0.0.0:5052/v1/proxy/unknown_service/valid_path", "404: Not found"),
        (
            "http://0.0.0.0:5052/v1/proxy/test_service/invalid_path/test_service/valid_path",
            "404: Not found",
        ),
    ],
)
async def test_request_url(
    async_client_factory,
    concurrency_limit,
    request_factory,
    request_url,
    expected_error,
):
    proxy_client = TestProxyClient(async_client_factory(), concurrency_limit)

    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            await proxy_client.proxy(request_factory(request_url=request_url))
    else:
        response = await proxy_client.proxy(request_factory(request_url=request_url))
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_body", "expected_error"),
    [
        (b'{"model": "model1"}', None),
        (b"model is model1", "400: Invalid JSON"),
        (b"", "400: Invalid JSON"),
    ],
)
async def test_request_body(
    async_client_factory,
    concurrency_limit,
    request_factory,
    request_body,
    expected_error,
):
    proxy_client = TestProxyClient(async_client_factory(), concurrency_limit)

    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            await proxy_client.proxy(request_factory(request_body=request_body))
    else:
        response = await proxy_client.proxy(request_factory(request_body=request_body))
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_body", "expected_error"),
    [
        (b'{"model": "model1"}', None),
        (b'{"model": "expensive-model"}', "400: Unsupported model"),
        (b'{"different_key": "model1"}', "400: Unsupported model"),
        (b"{}", "400: Unsupported model"),
    ],
)
async def test_model_names(
    async_client_factory,
    concurrency_limit,
    request_factory,
    request_body,
    expected_error,
):
    proxy_client = TestProxyClient(async_client_factory(), concurrency_limit)

    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            await proxy_client.proxy(request_factory(request_body=request_body))
    else:
        response = await proxy_client.proxy(request_factory(request_body=request_body))
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_headers", "expected_headers"),
    [
        (
            {"Content-Type": "application/json"},
            {"Content-Type": "application/json", "X-Test-Header": "test"},
        ),
        (
            {"Content-Type": "application/json", "X-Gitlab-Instance-Id": "123"},
            {"Content-Type": "application/json", "X-Test-Header": "test"},
        ),
        (
            {"Content-Type": "application/json", "X-Test-Header": "unknown"},
            {"Content-Type": "application/json", "X-Test-Header": "test"},
        ),
        (
            {},
            {"X-Test-Header": "test"},
        ),
    ],
)
async def test_upstream_headers(
    async_client_factory,
    concurrency_limit,
    request_factory,
    request_headers,
    expected_headers,
):
    async_client = async_client_factory()
    proxy_client = TestProxyClient(async_client, concurrency_limit)

    await proxy_client.proxy(request_factory(request_headers=request_headers))

    async_client.build_request.assert_called_once_with(
        "POST",
        URL("/valid_path"),
        headers=expected_headers,
        json={"model": "model1"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_body", "expected_streaming"),
    [
        (b'{"model": "model1"}', False),
        (b'{"model": "model1", "stream": true}', True),
    ],
)
async def test_streaming(
    async_client_factory,
    concurrency_limit,
    request_factory,
    request_body,
    expected_streaming,
):
    async_client = async_client_factory()
    proxy_client = TestProxyClient(async_client, concurrency_limit)

    response = await proxy_client.proxy(request_factory(request_body=request_body))

    async_client.send.assert_called_once_with(ANY, stream=expected_streaming)

    if expected_streaming:
        assert isinstance(response, fastapi.responses.StreamingResponse)
    else:
        assert isinstance(response, fastapi.Response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_headers", "expected_headers"),
    [
        (
            {},
            [("content-length", "22")],
        ),
        (
            {"Vendor-Trace-ID": "123"},
            [("content-length", "22")],
        ),
        (
            {"content-length": "200"},
            [("content-length", "200")],
        ),
    ],
)
async def test_downstream_headers(
    async_client_factory,
    concurrency_limit,
    request_factory,
    response_headers,
    expected_headers,
):
    async_client = async_client_factory(response_headers=response_headers)
    proxy_client = TestProxyClient(async_client, concurrency_limit)

    response = await proxy_client.proxy(request_factory())

    assert response.headers.items() == expected_headers
