from typing import AsyncIterator

from anthropic import AsyncAnthropic
from dependency_injector import containers, providers

from ai_gateway.models.v2.anthropic_claude import ChatAnthropic

__all__ = [
    "ContainerModels",
]


async def _init_anthropic_client() -> AsyncIterator[AsyncAnthropic]:
    async_client = AsyncAnthropic()

    yield async_client

    await async_client.close()


class ContainerModels(containers.DeclarativeContainer):
    # We need to resolve the model based on the model name provided by the upstream container.
    # Hence, `ChatAnthropic` etc. are only partially applied here.

    config = providers.Configuration(strict=True)

    http_async_client_anthropic = providers.Resource(_init_anthropic_client)

    anthropic_claude_chat_fn = providers.Factory(
        ChatAnthropic,
        async_client=http_async_client_anthropic,
    )
