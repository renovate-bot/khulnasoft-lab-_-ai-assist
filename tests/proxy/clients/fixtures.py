from unittest.mock import AsyncMock, Mock

import fastapi
import httpx
import pytest
from starlette.datastructures import URL

from ai_gateway.config import ConfigModelConcurrency


@pytest.fixture
def async_client_factory():
    def create(
        response_status_code: int = 200,
        response_headers: dict = {
            "Content-Type": "application/json",
            "date": "2024",
            "transfer-encoding": "chunked",
        },
        response_json: dict = {"response": "mocked"},
    ):
        client = Mock(spec=httpx.AsyncClient)
        client.send.return_value = httpx.Response(
            status_code=response_status_code,
            headers=response_headers,
            json=response_json,
        )
        return client

    return create


@pytest.fixture
def concurrency_limit():
    concurrency_limit = Mock(spec=ConfigModelConcurrency)
    concurrency_limit.for_model.return_value = 100
    return concurrency_limit


@pytest.fixture
def request_factory():
    def create(
        request_url: str = "http://0.0.0.0:5052/v1/proxy/test_service/valid_path",
        request_body: bytes = b'{"model": "model1"}',
        request_headers: dict = {"Content-Type": "application/json"},
    ):
        request = Mock(spec=fastapi.Request)
        request.url = URL(request_url)
        request.method = "POST"
        mock_request_body = AsyncMock()
        mock_request_body.return_value = request_body
        request.body = mock_request_body
        request.headers = request_headers
        return request

    return create
