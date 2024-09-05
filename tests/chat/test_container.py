from typing import cast

from dependency_injector import containers, providers

from ai_gateway.chat.agents import AdditionalContext, Context, ReActAgentInputs
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


def test_react_agent_inputs_with_additional_context():
    additional_context = AdditionalContext(
        type="file",
        name="research.py",
        content="This is some additional context about Italy",
    )

    inputs = ReActAgentInputs(
        question="What is the capital of Italy?",
        chat_history=[],
        agent_scratchpad=[],
        additional_context=[additional_context],
    )
    # pylint: disable=unsubscriptable-object
    assert isinstance(inputs.additional_context[0], AdditionalContext)
    # pylint: enable=unsubscriptable-object


def test_react_agent_inputs_without_additional_context():
    inputs = ReActAgentInputs(
        question="What is the capital of Italy?",
        chat_history=[],
        agent_scratchpad=[],
    )
    assert inputs.additional_context is None


def test_react_agent_inputs_with_context():
    context = Context(
        type="issue",
        content="This is an incredibly interesting issue",
    )

    inputs = ReActAgentInputs(
        question="What is the capital of Italy?",
        chat_history=[],
        agent_scratchpad=[],
        context=context,
    )
    assert isinstance(inputs.context, Context)


def test_react_agent_inputs_without_context():
    inputs = ReActAgentInputs(
        question="What is the capital of Italy?",
        chat_history=[],
        agent_scratchpad=[],
    )
    assert inputs.context is None
