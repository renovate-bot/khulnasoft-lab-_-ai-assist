from typing import Any, List, Mapping, Optional, Self

from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic as _LChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import model_validator

__all__ = ["ChatAnthropic"]


class ChatAnthropic(_LChatAnthropic):
    """
    A wrapper around `langchain_anthropic.ChatAnthropic`
    that accepts the Anthropic asynchronous client as an input parameter
    """

    async_client: AsyncAnthropic
    """Anthropic async HTTP client"""

    default_request_timeout: float | None = 60.0
    """Timeout for requests to Anthropic Completion API."""

    # sdk default = 2: https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#retries
    max_retries: int = 1
    """Number of retries allowed for requests sent to the Anthropic Completion API."""

    default_headers: Mapping[str, str] = {"anthropic-version": "2023-06-01"}
    """Headers to pass to the Anthropic clients, will be used for every API call."""

    @model_validator(mode="after")
    def post_init(self) -> Self:
        client_options = {}

        client_options["api_key"] = self.anthropic_api_key.get_secret_value()
        client_options["base_url"] = self.anthropic_api_url

        client_options.update(
            {
                "max_retries": self.max_retries,
                "default_headers": self.default_headers or None,
            }
        )

        # value <= 0 indicates the param should be ignored. None is a meaningful value
        # for Anthropic client and treated differently than not specifying the param at
        # all.
        if self.default_request_timeout is None or self.default_request_timeout > 0:
            client_options["timeout"] = self.default_request_timeout

        # pylint: disable=attribute-defined-outside-init
        async_client: AsyncAnthropic = self.async_client
        self._async_client = async_client.with_options(**client_options)

        # hack: we don't use sync methods in the AIGW,
        # so to avoid unnecessary initialization, set None for the sync client
        self._client = None
        # pylint: enable=attribute-defined-outside-init

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError()
