from typing import cast

from dependency_injector import containers, providers

from ai_gateway.models.anthropic import AnthropicModel


def test_container(mock_container: containers.DeclarativeContainer):
    x_ray = cast(providers.Container, mock_container.x_ray)

    assert isinstance(x_ray.anthropic_claude(), AnthropicModel)
