import httpx
from anthropic import AsyncAnthropic
from dependency_injector import containers, providers
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient

from ai_gateway.config import ConfigModelConcurrency
from ai_gateway.models import mock
from ai_gateway.models.anthropic import AnthropicChatModel, AnthropicModel
from ai_gateway.models.base import connect_anthropic, grpc_connect_vertex
from ai_gateway.models.litellm import LiteLlmChatModel, LiteLlmTextGenModel
from ai_gateway.models.vertex_text import (
    PalmCodeBisonModel,
    PalmCodeGeckoModel,
    PalmTextBisonModel,
)
from ai_gateway.proxy.clients import AnthropicProxyClient, VertexAIProxyClient

__all__ = [
    "ContainerModels",
]


def _init_vertex_grpc_client(
    endpoint: str,
    mock_model_responses: bool,
    custom_models_enabled: bool,
) -> PredictionServiceAsyncClient | None:
    if mock_model_responses or custom_models_enabled:
        return None

    return grpc_connect_vertex({"api_endpoint": endpoint})


def _init_anthropic_client(
    mock_model_responses: bool,
) -> AsyncAnthropic | None:
    if mock_model_responses:
        return None

    return connect_anthropic()


def _init_anthropic_proxy_client(
    mock_model_responses: bool,
) -> httpx.AsyncClient | mock.AsyncClient:
    if mock_model_responses:
        return mock.AsyncClient()

    return httpx.AsyncClient(
        base_url="https://api.anthropic.com/", timeout=httpx.Timeout(timeout=60.0)
    )


def _init_vertex_ai_proxy_client(
    mock_model_responses: bool,
    endpoint: str,
) -> httpx.AsyncClient | None:
    if mock_model_responses:
        return None

    return httpx.AsyncClient(
        base_url=f"https://{endpoint}/",
        timeout=httpx.Timeout(timeout=60.0),
    )


class ContainerModels(containers.DeclarativeContainer):
    # We need to resolve the model based on the model name provided by the upstream container.
    # Hence, `VertexTextBaseModel.from_model_name` and `AnthropicModel.from_model_name` are only partially applied here.

    config = providers.Configuration(strict=True)

    _mock_selector = providers.Callable(
        lambda mock_model_responses: "mocked" if mock_model_responses else "original",
        config.mock_model_responses,
    )

    grpc_client_vertex = providers.Singleton(
        _init_vertex_grpc_client,
        endpoint=config.vertex_text_model.endpoint,
        mock_model_responses=config.mock_model_responses,
        custom_models_enabled=config.custom_models.enabled,
    )

    http_client_anthropic = providers.Singleton(
        _init_anthropic_client,
        mock_model_responses=config.mock_model_responses,
    )

    http_client_anthropic_proxy = providers.Singleton(
        _init_anthropic_proxy_client,
        mock_model_responses=config.mock_model_responses,
    )

    http_client_vertex_ai_proxy = providers.Singleton(
        _init_vertex_ai_proxy_client,
        mock_model_responses=config.mock_model_responses,
        endpoint=config.vertex_text_model.endpoint,
    )

    vertex_text_bison = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            PalmTextBisonModel.from_model_name,
            client=grpc_client_vertex,
            project=config.vertex_text_model.project,
            location=config.vertex_text_model.location,
        ),
        mocked=providers.Factory(mock.LLM),
    )

    vertex_code_bison = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            PalmCodeBisonModel.from_model_name,
            client=grpc_client_vertex,
            project=config.vertex_text_model.project,
            location=config.vertex_text_model.location,
        ),
        mocked=providers.Factory(mock.LLM),
    )

    vertex_code_gecko = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            PalmCodeGeckoModel.from_model_name,
            client=grpc_client_vertex,
            project=config.vertex_text_model.project,
            location=config.vertex_text_model.location,
        ),
        mocked=providers.Factory(mock.LLM),
    )

    anthropic_claude = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            AnthropicModel.from_model_name, client=http_client_anthropic
        ),
        mocked=providers.Factory(mock.LLM),
    )

    anthropic_claude_chat = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            AnthropicChatModel.from_model_name,
            client=http_client_anthropic,
        ),
        mocked=providers.Factory(mock.ChatModel),
    )

    llmlite = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            LiteLlmTextGenModel.from_model_name,
            custom_models_enabled=config.custom_models.enabled,
        ),
        mocked=providers.Factory(mock.ChatModel),
    )

    llmlite_chat = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            LiteLlmChatModel.from_model_name,
            custom_models_enabled=config.custom_models.enabled,
        ),
        mocked=providers.Factory(mock.ChatModel),
    )

    anthropic_proxy_client = providers.Factory(
        AnthropicProxyClient,
        client=http_client_anthropic_proxy,
        concurrency_limit=providers.Factory(
            ConfigModelConcurrency, config.model_engine_concurrency_limits
        ),
    )

    vertex_ai_proxy_client = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            VertexAIProxyClient,
            client=http_client_vertex_ai_proxy,
            project=config.vertex_text_model.project,
            location=config.vertex_text_model.location,
            concurrency_limit=providers.Factory(
                ConfigModelConcurrency, config.model_engine_concurrency_limits
            ),
        ),
        mocked=providers.Factory(mock.ProxyClient),
    )
