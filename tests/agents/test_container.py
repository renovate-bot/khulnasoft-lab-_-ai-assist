from typing import cast

from dependency_injector import containers, providers

from ai_gateway.agents.registry import LocalAgentRegistry


def test_container(mock_container: containers.DeclarativeContainer):
    agents = cast(providers.Container, mock_container.pkg_agents)

    assert isinstance(agents.agent_registry(), LocalAgentRegistry)
