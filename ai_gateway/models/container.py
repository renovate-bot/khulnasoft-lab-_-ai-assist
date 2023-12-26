from typing import Callable, Iterator, Optional

from anthropic import AsyncAnthropic
from dependency_injector import containers, providers
from google.cloud.aiplatform.gapic import PredictionServiceAsyncClient

from ai_gateway.config import Config
from ai_gateway.models.anthropic import AnthropicModel
from ai_gateway.models.base import connect_anthropic, grpc_connect_vertex
from ai_gateway.models.fake import FakePalmTextGenModel
from ai_gateway.models.vertex_text import (
    PalmCodeBisonModel,
    PalmCodeGeckoModel,
    PalmTextBisonModel,
)

__all__ = [
    "ContainerModels",
]


def _real_or_fake(use_fake: bool) -> Callable:
    def _fn():
        return "fake" if use_fake else "real"

    return _fn


def _init_vertex_grpc_client(
    endpoint: str, use_fake: bool
) -> Iterator[Optional[PredictionServiceAsyncClient]]:
    if use_fake:
        yield None
        return

    client = grpc_connect_vertex({"api_endpoint": endpoint})
    yield client
    client.transport.close()


def _init_anthropic_client(use_fake: bool) -> Iterator[Optional[AsyncAnthropic]]:
    if use_fake:
        yield None
        return

    client = connect_anthropic()
    yield client
    client.close()


class ContainerModels(containers.DeclarativeContainer):
    # We need to resolve the model based on the model name provided in the request payload.
    # Hence, `VertexTextBaseModel.from_model_name` and `AnthropicModel.from_model_name` are only partially applied here.

    config = providers.Configuration()
    config.from_dict(Config().model_dump())

    grpc_client_vertex = providers.Resource(
        _init_vertex_grpc_client,
        endpoint=config.vertex_text_model.endpoint,
        use_fake=config.use_fake_models,
    )

    client_anthropic = providers.Resource(connect_anthropic)

    vertex_text_bison = providers.Selector(
        _real_or_fake(config.use_fake_models()),
        real=providers.Factory(
            PalmTextBisonModel.from_model_name,
            client=grpc_client_vertex,
            project=config.vertex_text_model.project,
            location=config.vertex_text_model.location,
        ),
        fake=providers.Factory(FakePalmTextGenModel),
    )

    vertex_code_bison = providers.Selector(
        _real_or_fake(config.use_fake_models()),
        real=providers.Factory(
            PalmCodeBisonModel.from_model_name,
            client=grpc_client_vertex,
            project=config.vertex_text_model.project,
            location=config.vertex_text_model.location,
        ),
        fake=providers.Factory(FakePalmTextGenModel),
    )

    vertex_code_gecko = providers.Selector(
        _real_or_fake(config.use_fake_models()),
        real=providers.Factory(
            PalmCodeGeckoModel.from_model_name,
            client=grpc_client_vertex,
            project=config.vertex_text_model.project,
            location=config.vertex_text_model.location,
        ),
        fake=providers.Factory(FakePalmTextGenModel),
    )

    anthropic_claude = providers.Selector(
        _real_or_fake(config.use_fake_models()),
        real=providers.Factory(
            AnthropicModel.from_model_name,
            client=client_anthropic,
        ),
        # TODO: We need to update our fake models to make them generic
        fake=providers.Factory(FakePalmTextGenModel),
    )
