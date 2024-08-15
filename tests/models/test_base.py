from unittest.mock import MagicMock, patch

import pytest
from anthropic import AsyncAnthropic
from httpx import AsyncClient, Limits

from ai_gateway.models.base import connect_anthropic


@pytest.mark.asyncio
async def test_connect_anthropic():
    with patch("ai_gateway.models.base._DefaultAsyncHttpxClient") as mock_client:
        mock_http_client = MagicMock(spec=AsyncClient)
        mock_client.return_value = mock_http_client

        client = connect_anthropic()

        assert isinstance(client, AsyncAnthropic)
        mock_client.assert_called_once()

        limits_arg = mock_client.call_args[1]["limits"]
        assert isinstance(limits_arg, Limits)
        assert limits_arg.max_connections == 1000
        assert limits_arg.max_keepalive_connections == 100
        assert limits_arg.keepalive_expiry == 30
