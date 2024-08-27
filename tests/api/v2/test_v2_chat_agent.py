from typing import AsyncIterator
from unittest.mock import Mock, PropertyMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from starlette.testclient import TestClient

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
from ai_gateway.chat.agents.react import ReActAgentFinalAnswer
from ai_gateway.config import Config
from ai_gateway.gitlab_features import WrongUnitPrimitives
from ai_gateway.prompts.typing import ModelMetadata


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


@pytest.fixture
def mock_model(model: BaseChatModel):
    with patch("ai_gateway.prompts.Prompt._build_model", return_value=model) as mock:
        yield mock


@pytest.fixture()
def mocked_tools():
    with patch(
        "ai_gateway.chat.executor.GLAgentRemoteExecutor.tools",
        new_callable=PropertyMock,
        return_value=[],
    ) as mock:
        yield mock


@pytest.fixture()
def mocked_on_behalf():
    def _on_behalf(user: GitLabUser, gl_version: str) -> AsyncIterator[TypeAgentAction]:
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


@pytest.fixture
def mock_config():
    config = Config()
    config.custom_models.enabled = True

    yield config


class TestReActAgentStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "prompt",
            "agent_options",
            "actions",
            "model_metadata",
            "expected_actions",
            "unavailable_resources",
        ),
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
                ["Mystery Resource 1", "Mystery Resource 2"],
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
        unavailable_resources: list[str],
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
                "unavailable_resources": unavailable_resources,
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
            unavailable_resources=unavailable_resources,
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


class TestChatAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "question",
            "agent_options",
            "model_response",
            "expected_actions",
        ),
        [
            (
                "Basic request",
                AgentRequestOptions(
                    chat_history="",
                    agent_scratchpad=ReActAgentScratchpad(agent_type="react", steps=[]),
                ),
                "thought\nFinal Answer: answer\n",
                [
                    AgentStreamResponseEvent(
                        type="final_answer_delta",
                        data=ReActAgentFinalAnswer(
                            text=c,
                        ),
                    )
                    for c in "answer"
                ],
            ),
            (
                "Request with '{}' in the input",
                AgentRequestOptions(
                    chat_history="",
                    agent_scratchpad=ReActAgentScratchpad(agent_type="react", steps=[]),
                    current_file=CurrentFile(
                        file_path="main.c",
                        data="int main() {}",
                        selected_code=True,
                    ),
                ),
                "thought\nFinal Answer: answer\n",
                [
                    AgentStreamResponseEvent(
                        type="final_answer_delta",
                        data=ReActAgentFinalAnswer(
                            text=c,
                        ),
                    )
                    for c in "answer"
                ],
            ),
        ],
    )
    async def test_request(
        self,
        mock_client: TestClient,
        mock_model: Mock,
        mocked_tools: Mock,
        mock_track_internal_event,
        question: str,
        agent_options: AgentRequestOptions,
        expected_actions: list[AgentStreamResponseEvent],
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "prompt": question,
                "options": agent_options.model_dump(mode="json"),
            },
        )

        actual_actions = [
            AgentStreamResponseEvent.model_validate_json(chunk)
            for chunk in response.text.strip().split("\n")
        ]

        assert response.status_code == 200
        assert actual_actions == expected_actions

        mock_track_internal_event.assert_called_once_with(
            "request_duo_chat",
            category="ai_gateway.api.v2.chat.agent",
        )
