from typing import AsyncIterator, Optional
from unittest.mock import Mock, patch

import pytest
from starlette.testclient import TestClient

from ai_gateway.agents.typing import ModelMetadata
from ai_gateway.api.v2 import api_router
from ai_gateway.api.v2.chat.typing import (
    AgentRequestOptions,
    AgentStreamResponseEvent,
    ReActAgentScratchpad,
)
from ai_gateway.auth import GitLabUser, User, UserClaims
from ai_gateway.chat.agents import (
    AgentStep,
    Context,
    CurrentFile,
    ReActAgentInputs,
    ReActAgentToolAction,
    TypeAgentAction,
    TypeReActAgentAction,
)
from ai_gateway.gitlab_features import WrongUnitPrimitives
from ai_gateway.internal_events import InternalEventAdditionalProperties


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
def mocked_stream():
    with patch("ai_gateway.chat.executor.GLAgentRemoteExecutor.stream") as mock:
        yield mock


@pytest.fixture()
def mocked_on_behalf():
    def _on_behalf(user: GitLabUser) -> AsyncIterator[TypeAgentAction]:
        if len(user.unit_primitives) == 0:
            # We don't expect any unit primitives allocated by the user
            raise WrongUnitPrimitives()
        else:
            raise Exception("raised exception to catch broken tests")

    with patch(
        "ai_gateway.chat.executor.GLAgentRemoteExecutor.on_behalf",
        side_effect=_on_behalf,
    ) as mock:
        yield mock


class TestReActAgentStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("prompt", "agent_options", "actions", "model_metadata", "expected_actions"),
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
                    current_file=CurrentFile(
                        file_path="main.py",
                        data="def main()",
                        selected_code=True,
                    ),
                ),
                [
                    ReActAgentToolAction(
                        thought="thought",
                        tool="tool",
                        tool_input="tool_input",
                    )
                ],
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint="http://localhost:4000",
                    api_key="token",
                ),
                [
                    AgentStreamResponseEvent(
                        type="action",
                        data=ReActAgentToolAction(
                            thought="thought",
                            tool="tool",
                            tool_input="tool_input",
                        ),
                    )
                ],
            )
        ],
    )
    async def test_success(
        self,
        mock_client: TestClient,
        mocked_stream: Mock,
        mock_track_internal_event,
        prompt: str,
        agent_options: AgentRequestOptions,
        actions: list[TypeAgentAction],
        expected_actions: list[AgentStreamResponseEvent],
        model_metadata: ModelMetadata,
    ):
        async def _agent_stream(*_args, **_kwargs) -> AsyncIterator[TypeAgentAction]:
            for action in actions:
                yield action

        mocked_stream.side_effect = _agent_stream

        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "prompt": prompt,
                "options": agent_options.model_dump(mode="json"),
                "model_metadata": model_metadata.model_dump(mode="json"),
            },
        )

        actual_actions = [
            AgentStreamResponseEvent.model_validate_json(chunk)
            for chunk in response.text.strip().split("\n")
        ]

        agent_scratchpad = [
            AgentStep[TypeReActAgentAction](
                action=ReActAgentToolAction(
                    thought=step.thought,
                    tool=step.tool,
                    tool_input=step.tool_input,
                ),
                observation=step.observation,
            )
            for step in agent_options.agent_scratchpad.steps
        ]

        agent_inputs = ReActAgentInputs(
            question=prompt,
            chat_history=agent_options.chat_history,
            agent_scratchpad=agent_scratchpad,
            context=agent_options.context,
            current_file=agent_options.current_file,
            model_metadata=model_metadata,
        )

        assert response.status_code == 200
        assert actual_actions == expected_actions
        mocked_stream.assert_called_once_with(inputs=agent_inputs)

        mock_track_internal_event.assert_called_once_with(
            "request_duo_chat",
            category="ai_gateway.api.v2.chat.agent",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "auth_user",
        [(User(authenticated=True, claims=UserClaims(scopes="wrong_scope")))],
    )
    async def test_exception_403(
        self,
        auth_user: User,
        mock_client: TestClient,
        mocked_on_behalf: Mock,
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "prompt": "random prompt",
                "options": AgentRequestOptions(
                    chat_history="chat history",
                    agent_scratchpad=ReActAgentScratchpad(
                        agent_type="react",
                        steps=[],
                    ),
                ).model_dump(mode="json"),
            },
        )

        assert response.status_code == 403
        mocked_on_behalf.assert_called_once()
