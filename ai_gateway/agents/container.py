from dependency_injector import containers, providers

from ai_gateway.agents.chat import ReActAgent
from ai_gateway.agents.registry import Key, LocalAgentRegistry, ModelProvider

__all__ = [
    "ContainerAgents",
]


class ContainerAgents(containers.DeclarativeContainer):
    models = providers.DependenciesContainer()

    _anthropic_claude_fn = providers.Factory(models.anthropic_claude_chat_fn)

    agent_registry = providers.Singleton(
        LocalAgentRegistry.from_local_yaml,
        data={Key(use_case="chat", type="react"): ReActAgent},
        model_factories={ModelProvider.ANTHROPIC: _anthropic_claude_fn},
    )
