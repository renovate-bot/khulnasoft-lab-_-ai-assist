from dependency_injector import containers, providers
from langchain_community.chat_models import ChatLiteLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from ai_gateway.models import mock
from ai_gateway.models.base import init_anthropic_client, log_request
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.prompts.typing import Model

__all__ = [
    "ContainerModels",
]


def _litellm_factory(*args, **kwargs) -> Model:
    model = ChatLiteLLM(*args, **kwargs)

    if kwargs.get("custom_llm_provider", "") == "vertex_ai":
        client = AsyncHTTPHandler(event_hooks={"request": [log_request]})

        return model.bind(client=client)

    return model


class ContainerModels(containers.DeclarativeContainer):
    # We need to resolve the model based on the model name provided by the upstream container.
    # Hence, `ChatAnthropic` etc. are only partially applied here.

    config = providers.Configuration(strict=True)

    _mock_selector = providers.Callable(
        lambda mock_model_responses: "mocked" if mock_model_responses else "original",
        config.mock_model_responses,
    )

    http_async_client_anthropic = providers.Singleton(
        init_anthropic_client,
        mock_model_responses=config.mock_model_responses,
    )

    anthropic_claude_chat_fn = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            ChatAnthropic,
            async_client=http_async_client_anthropic,
        ),
        mocked=providers.Factory(mock.FakeModel),
    )

    lite_llm_chat_fn = providers.Factory(_litellm_factory)
