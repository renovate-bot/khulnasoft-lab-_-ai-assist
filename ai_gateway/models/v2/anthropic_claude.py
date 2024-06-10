from typing import Any, List, Mapping, Optional

import httpx
from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic as _LChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import convert_to_secret_str

__all__ = ["ChatAnthropic"]


class ChatAnthropic(_LChatAnthropic):
    """
    A wrapper around `langchain_anthropic.ChatAnthropic`
    that accepts the Anthropic asynchronous client as an input parameter
    """

    async_client: AsyncAnthropic
    """Anthropic async HTTP client"""

    default_request_timeout: float | httpx.Timeout | None = httpx.Timeout(
        60.0, connect=5.0
    )
    """Timeout for requests to Anthropic Completion API."""

    # sdk default = 2: https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#retries
    max_retries: int = 1
    """Number of retries allowed for requests sent to the Anthropic Completion API."""

    default_headers: Mapping[str, str] = {"anthropic-version": "2023-06-01"}
    """Headers to pass to the Anthropic clients, will be used for every API call."""

    @root_validator()
    # pylint: disable-next=no-self-argument
    def validate_environment(cls, values: dict) -> dict:
        client_options = {}

        if anthropic_api_key := values.get("anthropic_api_key", None):
            anthropic_api_key = convert_to_secret_str(anthropic_api_key)
            values["anthropic_api_key"] = anthropic_api_key
            client_options["api_key"] = anthropic_api_key.get_secret_value()

        if api_url := values.get("anthropic_api_url", None):
            client_options["base_url"] = api_url

        client_options.update(
            {
                "max_retries": values["max_retries"],
                "default_headers": values.get("default_headers"),
            }
        )

        # value <= 0 indicates the param should be ignored. None is a meaningful value
        # for Anthropic client and treated differently than not specifying the param at
        # all.
        if (
            values["default_request_timeout"] is None
            or isinstance(values["default_request_timeout"], httpx.Timeout)
            or values["default_request_timeout"] > 0
        ):
            client_options["timeout"] = values["default_request_timeout"]

        async_client: AsyncAnthropic = values["async_client"]
        values["_async_client"] = async_client.with_options(**client_options)

        # hack: we don't use sync methods in the AIGW,
        # so to avoid unnecessary initialization, set None for the sync client
        values["_client"] = None

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError()
