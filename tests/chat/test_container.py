from typing import cast

from dependency_injector import containers, providers

from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.models.anthropic import (
    AnthropicChatModel,
    AnthropicModel,
    KindAnthropicModel,
)
from ai_gateway.models.litellm import KindLiteLlmModel, LiteLlmChatModel


def test_container(mock_container: containers.DeclarativeContainer):
    chat = cast(providers.Container, mock_container.chat)

    assert isinstance(
        chat.anthropic_claude_factory("llm", name=KindAnthropicModel.CLAUDE_2_0),
        AnthropicModel,
    )
    assert isinstance(
        chat.anthropic_claude_factory("chat", name=KindAnthropicModel.CLAUDE_2_0),
        AnthropicChatModel,
    )
    assert isinstance(
        chat.litellm_factory(name=KindLiteLlmModel.MISTRAL), LiteLlmChatModel
    )
    assert isinstance(chat.gl_agent_remote_executor(), GLAgentRemoteExecutor)
