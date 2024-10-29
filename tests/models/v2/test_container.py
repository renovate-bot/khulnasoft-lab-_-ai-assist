from unittest import mock

from ai_gateway.models.base import log_request
from ai_gateway.models.v2.container import _litellm_factory


@mock.patch("ai_gateway.models.v2.container.ChatLiteLLM")
@mock.patch("ai_gateway.models.v2.container.AsyncHTTPHandler")
def test_litellm_factory(
    mock_async_http_handler: mock.Mock, mock_chat_lite_llm: mock.Mock
):
    client = mock.Mock()
    mock_async_http_handler.return_value = client
    model = mock.Mock()
    mock_chat_lite_llm.return_value = model
    binding = mock.Mock()
    model.bind.return_value = binding

    kwargs = {"model": "claude-3-sonnet@20240229", "custom_llm_provider": "vertex_ai"}

    assert _litellm_factory(**kwargs) == binding

    mock_chat_lite_llm.assert_called_once_with(**kwargs)
    mock_async_http_handler.assert_called_once_with(
        event_hooks={"request": [log_request]}
    )
    model.bind.assert_called_once_with(client=client)
