import json
from typing import AsyncIterator
from unittest.mock import Mock, PropertyMock, call, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from starlette.testclient import TestClient

from ai_gateway.api.v2 import api_router
from ai_gateway.api.v2.chat.typing import (
    AgentRequest,
    AgentRequestOptions,
    ReActAgentScratchpad,
)
from ai_gateway.auth import GitLabUser, User, UserClaims
from ai_gateway.chat.agents import (
    AdditionalContext,
    AgentBaseEvent,
    AgentStep,
    AgentToolAction,
    Context,
    CurrentFile,
    Message,
    ReActAgentInputs,
)
from ai_gateway.chat.agents.typing import AgentFinalAnswer, TypeAgentEvent
from ai_gateway.config import Config
from ai_gateway.gitlab_features import WrongUnitPrimitives
from ai_gateway.models.base_chat import Role
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


@pytest.fixture
def mock_config():
    config = Config()
    config.custom_models.enabled = True

    yield config


def chunk_to_model(chunk: str, klass: AgentBaseEvent) -> str:
    res = json.loads(chunk)
    data = res.pop("data")
    return klass.model_validate({**res, **data})


class TestReActAgentStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("agent_request", "model_response", "expected_events"),
        [
            (
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="How can I write hello world in python?",
                        ),
                    ]
                ),
                "thought\nFinal Answer: answer\n",
                [AgentFinalAnswer(text=c) for c in "answer"],
            ),
            (
                AgentRequest(
                    messages=[
                        Message(role=Role.USER, content="chat history"),
                        Message(role=Role.ASSISTANT, content="chat history"),
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                            context=Context(type="issue", content="issue content"),
                            current_file=CurrentFile(
                                file_path="main.py",
                                data="def main()",
                                selected_code=True,
                            ),
                        ),
                    ],
                    options=AgentRequestOptions(
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
                    ),
                    model_metadata=ModelMetadata(
                        name="mistral",
                        provider="litellm",
                        endpoint="http://localhost:4000",
                        api_key="token",
                    ),
                    unavailable_resources=["Mystery Resource 1", "Mystery Resource 2"],
                ),
                "thought\nFinal Answer: answer\n",
                [AgentFinalAnswer(text=c) for c in "answer"],
            ),
        ],
    )
    async def test_success(
        self,
        mock_client: TestClient,
        mock_model: Mock,
        mocked_tools: Mock,
        agent_request: AgentRequest,
        expected_events: list[TypeAgentEvent],
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json=agent_request.model_dump(mode="json"),
        )

        actual_events = [
            chunk_to_model(chunk, AgentFinalAnswer)
            for chunk in response.text.strip().split("\n")
        ]

        assert response.status_code == 200
        assert actual_events == expected_events

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
                    AgentToolAction(
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
                    AgentToolAction(
                        thought="thought",
                        tool="tool",
                        tool_input="tool_input",
                    )
                ],
                ["Mystery Resource 1", "Mystery Resource 2"],
            )
        ],
    )
    async def test_legacy_success(
        self,
        mock_client: TestClient,
        mocked_stream: Mock,
        mock_track_internal_event,
        prompt: str,
        agent_options: AgentRequestOptions,
        actions: list[TypeAgentEvent],
        expected_actions: list[TypeAgentEvent],
        model_metadata: ModelMetadata,
        unavailable_resources: list[str],
    ):
        async def _agent_stream(*_args, **_kwargs) -> AsyncIterator[TypeAgentEvent]:
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
            chunk_to_model(chunk, AgentToolAction)
            for chunk in response.text.strip().split("\n")
        ]

        agent_scratchpad = [
            AgentStep(
                action=AgentToolAction(
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
        "auth_user,agent_request,expected_status_code,expected_error,expected_internal_events",
        [
            (
                User(authenticated=True, claims=UserClaims(scopes=["duo_chat"])),
                AgentRequest(messages=[Message(role=Role.USER, content="Hi")]),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
            ),
            (
                User(authenticated=True, claims=UserClaims(scopes="wrong_scope")),
                AgentRequest(messages=[Message(role=Role.USER, content="Hi")]),
                403,
                '{"detail":"Unauthorized to access duo chat"}',
                [],
            ),
            (
                User(
                    authenticated=True,
                    claims=UserClaims(scopes=["duo_chat", "include_file_context"]),
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=[AdditionalContext(category="file")],
                        )
                    ]
                ),
                200,
                "",
                [
                    call(
                        "request_include_file_context",
                        category="ai_gateway.api.v2.chat.agent",
                    ),
                ],
            ),
            (
                User(authenticated=True, claims=UserClaims(scopes=["duo_chat"])),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=[],
                        )
                    ]
                ),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
            ),
            (
                User(authenticated=True, claims=UserClaims(scopes=["duo_chat"])),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=None,
                        )
                    ]
                ),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
            ),
            (
                User(authenticated=True, claims=UserClaims(scopes=["duo_chat"])),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=[AdditionalContext(category="file")],
                        )
                    ]
                ),
                403,
                '{"detail":"Unauthorized to access include_file_context"}',
                [],
            ),
        ],
    )
    async def test_authorization(
        self,
        auth_user: User,
        agent_request: AgentRequest,
        mock_client: TestClient,
        mock_model: Mock,
        expected_status_code: int,
        expected_error: str,
        expected_internal_events,
        mock_track_internal_event: Mock,
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json=agent_request.model_dump(mode="json"),
        )

        assert response.status_code == expected_status_code

        if expected_error:
            assert response.text == expected_error
        else:
            mock_track_internal_event.assert_has_calls(expected_internal_events)


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
                    AgentFinalAnswer(
                        text=c,
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
                    additional_context=[AdditionalContext(category="merge_request")],
                ),
                "thought\nFinal Answer: answer\n",
                [
                    AgentFinalAnswer(
                        text=c,
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
        expected_actions: list[TypeAgentEvent],
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
            chunk_to_model(chunk, AgentFinalAnswer)
            for chunk in response.text.strip().split("\n")
        ]

        assert response.status_code == 200
        assert actual_actions == expected_actions

        mock_track_internal_event.assert_called_once_with(
            "request_duo_chat",
            category="ai_gateway.api.v2.chat.agent",
        )
