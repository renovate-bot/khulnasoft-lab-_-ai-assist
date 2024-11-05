from typing import Type, cast

import pytest
from dependency_injector import containers, providers

from ai_gateway import Config
from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.chat.tools import BaseTool
from ai_gateway.chat.tools.gitlab import (
    GitlabDocumentation,
    SelfHostedGitlabDocumentation,
)
from ai_gateway.models.anthropic import (
    AnthropicChatModel,
    AnthropicModel,
    KindAnthropicModel,
)
from ai_gateway.models.litellm import KindLiteLlmModel, LiteLlmChatModel


@pytest.fixture
def mock_config(custom_models_enabled: bool):
    config = Config()
    config.custom_models.enabled = custom_models_enabled

    yield config


@pytest.mark.parametrize("custom_models_enabled", [False])
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


@pytest.mark.parametrize(
    ("custom_models_enabled", "expected_tool_type"),
    [(True, SelfHostedGitlabDocumentation), (False, GitlabDocumentation)],
)
def test_container_with_config(
    mock_container: containers.DeclarativeContainer,
    custom_models_enabled: bool,
    expected_tool_type: Type[BaseTool],
):
    chat = cast(providers.Container, mock_container.chat)

    tool_types = {type(tool) for tool in chat.gl_agent_remote_executor().tools}

    assert expected_tool_type in tool_types
