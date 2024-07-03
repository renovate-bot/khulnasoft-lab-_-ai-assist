from typing import Iterator

from anthropic import AsyncAnthropic
from dependency_injector import containers, providers
from langchain_community.chat_models import ChatLiteLLM

from ai_gateway.models.v2.anthropic_claude import ChatAnthropic

__all__ = [
    "ContainerModels",
]


def _init_anthropic_client() -> Iterator[AsyncAnthropic]:
    async_client = AsyncAnthropic()

    yield async_client

    async_client.close()


class ContainerModels(containers.DeclarativeContainer):
    # We need to resolve the model based on the model name provided by the upstream container.
    # Hence, `ChatAnthropic` etc. are only partially applied here.

    config = providers.Configuration(strict=True)

    http_async_client_anthropic = providers.Resource(_init_anthropic_client)

    anthropic_claude_chat_fn = providers.Factory(
        ChatAnthropic,
        async_client=http_async_client_anthropic,
    )

    lite_llm_chat_fn = providers.Factory(ChatLiteLLM)
