from dependency_injector import containers, providers

from ai_gateway.agents.config import ModelClassProvider
from ai_gateway.agents.registry import LocalAgentRegistry
from ai_gateway.chat import agents as chat

__all__ = [
    "ContainerAgents",
]


class ContainerAgents(containers.DeclarativeContainer):
    models = providers.DependenciesContainer()

    _anthropic_claude_fn = providers.Factory(models.anthropic_claude_chat_fn)
    _lite_llm_chat_fn = providers.Factory(models.lite_llm_chat_fn)

    agent_registry = providers.Singleton(
        LocalAgentRegistry.from_local_yaml,
        model_factories={
            ModelClassProvider.ANTHROPIC: _anthropic_claude_fn,
            ModelClassProvider.LITE_LLM: _lite_llm_chat_fn,
        },
        class_overrides={"chat/react": chat.ReActAgent},
    )
