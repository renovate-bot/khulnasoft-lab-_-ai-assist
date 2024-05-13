from typing import AsyncIterator
from unittest import mock

import pytest
from starlette.testclient import TestClient

from ai_gateway.api.v2 import api_router
from ai_gateway.api.v2.chat.typing import (
    AgentRequestOptions,
    AgentStreamResponseEvent,
    ReActAgentScratchpad,
)
from ai_gateway.auth import User, UserClaims
from ai_gateway.chat import GLAgentRemoteExecutor
from ai_gateway.chat.agents import AgentStep, TypeAgentAction
from ai_gateway.chat.agents.react import ReActAgentInputs, ReActAgentToolAction
from ai_gateway.chat.typing import Context
from ai_gateway.container import ContainerApplication


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(scopes=["duo_chat"]),
    )


@pytest.fixture()
def mocked_react_executor():
    container = ContainerApplication()

    mocked_executor = mock.Mock(spec=GLAgentRemoteExecutor)
    with container.chat.gl_agent_remote_executor.override(mocked_executor):
        yield mocked_executor


class TestReActAgentStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("prompt", "agent_options", "actions", "expected_actions"),
        [
            (
                "What's the title of this issue?",
                AgentRequestOptions(
                    chat_history="chat history",
                    agent_scratchpad=ReActAgentScratchpad(
                        agent_type="react",
                        steps=[
                            ReActAgentScratchpad.AgentStep(
                                thought="thought",
                                tool="tool",
                                tool_input="tool_input",
                                observation="observation",
                            )
                        ],
                    ),
                    context=Context(type="issue", content="issue content"),
                ),
                [
                    ReActAgentToolAction(
                        thought="thought",
                        tool="tool",
                        tool_input="tool_input",
                        log="log",
                    )
                ],
                [
                    AgentStreamResponseEvent(
                        type="action",
                        data=ReActAgentToolAction(
                            thought="thought",
                            tool="tool",
                            tool_input="tool_input",
                            log="log",
                        ),
                    )
                ],
            )
        ],
    )
    async def test_success(
        self,
        mock_client: TestClient,
        mocked_react_executor: mock.Mock,
        prompt: str,
        agent_options: AgentRequestOptions,
        actions: list[TypeAgentAction],
        expected_actions: list[AgentStreamResponseEvent],
    ):
        async def _agent_stream(*_args, **_kwargs) -> AsyncIterator[TypeAgentAction]:
            for action in actions:
                yield action

        mocked_react_executor.stream = mock.Mock(side_effect=_agent_stream)

        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "prompt": prompt,
                "options": agent_options.model_dump(mode="json"),
            },
        )

        actual_actions = [
            AgentStreamResponseEvent.model_validate_json(chunk)
            for chunk in response.text.strip().split("\n")
        ]

        agent_inputs = ReActAgentInputs(
            question=prompt,
            chat_history=agent_options.chat_history,
            context=agent_options.context,
        )

        agent_scratchpad = [
            AgentStep(
                action=ReActAgentToolAction(
                    thought=step.thought,
                    tool=step.tool,
                    tool_input=step.tool_input,
                ),
                observation=step.observation,
            )
            for step in agent_options.agent_scratchpad.steps
        ]

        assert response.status_code == 200
        assert actual_actions == expected_actions
        mocked_react_executor.stream.assert_called_with(
            inputs=agent_inputs, scratchpad=agent_scratchpad
        )
