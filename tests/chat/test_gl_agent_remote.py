from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.runnables import Runnable

from ai_gateway.auth import GitLabUser
from ai_gateway.chat import BaseToolsRegistry, GLAgentRemoteExecutor, TypeAgentFactory
from ai_gateway.chat.agents import ReActAgentInputs, ReActAgentToolAction
from ai_gateway.chat.tools.gitlab import EpicReader, IssueReader

expected_tool_action = ReActAgentToolAction(
    thought="thought", tool="tool", tool_input="tool_input"
)

expected_tools = [IssueReader(), EpicReader()]


@pytest.fixture()
def agent():
    async def _stream_agent(*_args, **_kwargs):
        for action in [expected_tool_action]:
            yield action

    agent = Mock(spec=Runnable)
    agent.ainvoke = AsyncMock(
        side_effect=lambda *_args, **_kwargs: expected_tool_action
    )
    agent.astream = Mock(side_effect=_stream_agent)

    return agent


@pytest.fixture()
def agent_factory(agent):
    agent_factory = Mock(
        spec=TypeAgentFactory, side_effect=lambda *_args, **_kwargs: agent
    )

    return agent_factory


@pytest.fixture()
def tools_registry():
    tools_registry = Mock(spec=BaseToolsRegistry)
    tools_registry.get_all = Mock(side_effect=lambda: expected_tools)
    tools_registry.get_on_behalf = Mock(side_effect=lambda _user: expected_tools)

    return tools_registry


@pytest.mark.parametrize(
    ("inputs", "user"),
    [
        (
            ReActAgentInputs(
                question="debug question",
                chat_history="debug chat_history",
                agent_scratchpad=[],
            ),
            GitLabUser(authenticated=True, is_debug=True),
        ),
        (
            ReActAgentInputs(
                question="question",
                chat_history="chat_history",
                agent_scratchpad=[],
            ),
            GitLabUser(authenticated=True, is_debug=False),
        ),
    ],
)
class TestGLAgentRemoteExecutor:
    @pytest.mark.asyncio
    async def test_invoke(
        self,
        agent: Mock,
        agent_factory: Mock,
        tools_registry: Mock,
        inputs: ReActAgentInputs,
        user: GitLabUser,
    ):
        executor = GLAgentRemoteExecutor(
            agent_factory=agent_factory, tools_registry=tools_registry
        )

        executor.on_behalf(user)

        actual_action = await executor.invoke(inputs=inputs)

        if user.is_debug:
            tools_registry.get_all.assert_called_once_with()
        else:
            tools_registry.get_on_behalf.assert_called_once_with(user)

        agent_factory.assert_called_once_with(tools=expected_tools, inputs=inputs)
        agent.ainvoke.assert_called_once_with(inputs)
        assert actual_action == expected_tool_action

    @pytest.mark.asyncio
    async def test_stream(
        self,
        agent: Mock,
        agent_factory: Mock,
        tools_registry: Mock,
        inputs: ReActAgentInputs,
        user: GitLabUser,
    ):
        executor = GLAgentRemoteExecutor(
            agent_factory=agent_factory, tools_registry=tools_registry
        )

        executor.on_behalf(user)

        actual_actions = [action async for action in executor.stream(inputs=inputs)]

        if user.is_debug:
            tools_registry.get_all.assert_called_once_with()
        else:
            tools_registry.get_on_behalf.assert_called_once_with(user)

        agent_factory.assert_called_once_with(tools=expected_tools, inputs=inputs)
        agent.astream.assert_called_once_with(inputs)
        assert actual_actions == [expected_tool_action]
