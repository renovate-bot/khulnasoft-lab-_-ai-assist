import httpx
import pytest
from anthropic import Anthropic, AsyncAnthropic

from ai_gateway.models.v2 import ChatAnthropic


class TestChatAnthropic:
    @pytest.mark.parametrize(
        ("model_options", "expected_options"),
        [
            (
                {},
                {
                    "default_request_timeout": httpx.Timeout(60.0, connect=5.0),
                    "max_retries": 1,
                    "default_headers": {"anthropic-version": "2023-06-01"},
                },
            ),
            (
                {
                    "default_request_timeout": 10,
                    "max_retries": 2,
                    "default_headers": {"anthropic-version": "2021-06-01"},
                },
                {
                    "default_request_timeout": 10,
                    "max_retries": 2,
                    "default_headers": {"anthropic-version": "2021-06-01"},
                },
            ),
        ],
    )
    def test_async_model_options(self, model_options: dict, expected_options: dict):
        model = ChatAnthropic(
            async_client=AsyncAnthropic(), model="claude-2.1", **model_options
        )  # type: ignore[call-arg]

        assert isinstance(model._async_client, AsyncAnthropic)
        assert (
            model._async_client.timeout == expected_options["default_request_timeout"]
        )
        assert model._async_client.max_retries == expected_options["max_retries"]

        all_headers = [
            model._async_client.default_headers[h_key] == h_value
            for h_key, h_value in expected_options["default_headers"].items()
        ]
        assert all(all_headers)

    def test_unsupported_sync_methods(self):
        model = ChatAnthropic(
            async_client=AsyncAnthropic(), model="claude-2.1"
        )  # type: ignore[call-arg]

        with pytest.raises(NotImplementedError):
            model.invoke("What's your name?")

    def test_overwrite_anthropic_credentials(self):
        model = ChatAnthropic(
            async_client=AsyncAnthropic(),
            model="claude-2.1",
            anthropic_api_key="test_api_key",
            anthropic_api_url="http://anthropic.test",
        )  # type: ignore[call-arg]

        assert model.anthropic_api_key.get_secret_value() == "test_api_key"
        assert model._async_client.api_key == "test_api_key"

        assert model.anthropic_api_url == "http://anthropic.test"
        assert model._async_client.base_url == "http://anthropic.test"
