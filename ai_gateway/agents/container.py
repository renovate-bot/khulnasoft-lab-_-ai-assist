from dependency_injector import containers, providers

from ai_gateway.agents.config import ModelClassProvider
from ai_gateway.agents.registry import LocalAgentRegistry
from ai_gateway.chat import agents as chat

__all__ = [
    "ContainerAgents",
]


class ContainerAgents(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)
    models = providers.DependenciesContainer()

    agent_registry = providers.Singleton(
        LocalAgentRegistry.from_local_yaml,
        class_overrides={
            "chat/react": chat.ReActAgent,
        },
        model_factories={
            ModelClassProvider.ANTHROPIC: providers.Factory(
                models.anthropic_claude_chat_fn
            ),
            ModelClassProvider.LITE_LLM: providers.Factory(models.lite_llm_chat_fn),
        },
        custom_models_enabled=config.custom_models.enabled,
    )
