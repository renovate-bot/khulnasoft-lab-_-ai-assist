from typing import cast

from dependency_injector import containers, providers

from ai_gateway.prompts.registry import LocalPromptRegistry


def test_container(mock_container: containers.DeclarativeContainer):
    prompts = cast(providers.Container, mock_container.pkg_prompts)

    assert isinstance(prompts.prompt_registry(), LocalPromptRegistry)
