from dependency_injector import containers, providers

from ai_gateway.agents.registry import LocalAgentRegistry

__all__ = [
    "ContainerAgents",
]


class ContainerAgents(containers.DeclarativeContainer):
    models = providers.DependenciesContainer()
    agent_registry = providers.Singleton(
        LocalAgentRegistry, client=models.http_client_anthropic
    )
