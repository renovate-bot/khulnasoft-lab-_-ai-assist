from typing import TYPE_CHECKING

from dependency_injector import containers, providers

from ai_gateway.chat.agents import ReActAgent, ReActAgentInputs, TypeAgentEvent
from ai_gateway.chat.executor import GLAgentRemoteExecutor, TypeAgentFactory
from ai_gateway.chat.toolset import DuoChatToolsRegistry
from ai_gateway.prompts.typing import ModelMetadata

if TYPE_CHECKING:
    from ai_gateway.prompts import BasePromptRegistry

__all__ = [
    "ContainerChat",
]


def _react_agent_factory(
    prompt_registry: "BasePromptRegistry",
) -> TypeAgentFactory[ReActAgentInputs, TypeAgentEvent]:
    def _fn(model_metadata: ModelMetadata) -> ReActAgent:
        return prompt_registry.get("chat/react", model_metadata)

    return _fn


class ContainerChat(containers.DeclarativeContainer):
    prompts = providers.DependenciesContainer()
    models = providers.DependenciesContainer()

    # The dependency injector does not allow us to override the FactoryAggregate provider directly.
    # However, we can still override its internal sub-factories to achieve the same goal.
    _anthropic_claude_llm_factory = providers.Factory(models.anthropic_claude)
    _anthropic_claude_chat_factory = providers.Factory(models.anthropic_claude_chat)

    _react_agent_factory = providers.Factory(
        _react_agent_factory,
        prompt_registry=prompts.prompt_registry,
    )

    # We need to resolve the model based on model name provided in request payload
    # Hence, `models._anthropic_claude` and `models._anthropic_claude_chat_factory` are only partially applied here.
    anthropic_claude_factory = providers.FactoryAggregate(
        llm=_anthropic_claude_llm_factory, chat=_anthropic_claude_chat_factory
    )

    litellm_factory = providers.Factory(models.litellm_chat)

    gl_agent_remote_executor = providers.Factory(
        GLAgentRemoteExecutor,
        agent_factory=_react_agent_factory,
        tools_registry=DuoChatToolsRegistry(),
    )
