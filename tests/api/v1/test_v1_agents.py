from typing import Any, List, Optional, Type
from unittest.mock import patch

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.agents import Agent
from ai_gateway.api.v1 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.chat.agents import ReActAgent
from ai_gateway.config import Config
from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.internal_events import InternalEventAdditionalProperties


class FakeModel(SimpleChatModel):
    expected_message: str
    response: str

    @property
    def _llm_type(self) -> str:
        return "fake-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        assert self.expected_message == messages[0].content

        return self.response


@pytest.fixture
def mock_agent_klass():
    yield Agent


@pytest.fixture
def mock_config():
    config = Config()
    config.custom_models.enabled = False

    yield config


@pytest.fixture
def mock_registry_get(mock_agent_klass: Optional[Type[Agent]]):
    with patch("ai_gateway.agents.registry.LocalAgentRegistry.get") as mock:
        if mock_agent_klass:
            model = FakeModel(
                expected_message="Hi, I'm John and I'm 20 years old",
                response="Hi John!",
            )

            mock.return_value = mock_agent_klass(
                name="fake_agent",
                chain=ChatPromptTemplate.from_messages(
                    ["Hi, I'm {name} and I'm {age} years old"]
                )
                | model,
                unit_primitives=[GitLabUnitPrimitive.EXPLAIN_VULNERABILITY],
            )
        else:
            mock.side_effect = KeyError()

        yield mock


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(
            scopes=[
                GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
            ]
        ),
    )


class TestAgent:
    @pytest.mark.parametrize(
        ("mock_agent_klass", "inputs", "expected_status", "expected_response"),
        [
            (Agent, {"name": "John", "age": 20}, 200, "Hi John!"),
            (
                None,
                {"name": "John", "age": 20},
                404,
                {"detail": "Agent 'test' not found"},
            ),
            (
                Agent,
                {"name": "John"},
                422,
                {
                    "detail": "\"Input to ChatPromptTemplate is missing variables {'age'}.  Expected: ['age', 'name'] Received: ['name']\""
                },
            ),
            (
                ReActAgent,
                {"name": "John", "age": 20},
                422,
                {"detail": "Agent 'test' is not supported"},
            ),
        ],
    )
    def test_request(
        self,
        mock_agent_klass,
        mock_registry_get,
        mock_client,
        mock_track_internal_event,
        inputs: dict[str, str],
        expected_status: int,
        expected_response: Any,
    ):
        response = mock_client.post(
            "/agents/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json=inputs,
        )

        mock_registry_get.assert_called_with("test", None, None)
        assert response.status_code == expected_status
        assert response.json() == expected_response

        if mock_agent_klass:
            mock_track_internal_event.assert_called_once_with(
                "request_explain_vulnerability",
                category="ai_gateway.api.v1.agents.invoke",
            )
        else:
            mock_track_internal_event.assert_not_called()


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(
        self, mock_container, mock_client, mock_registry_get
    ):
        response = mock_client.post(
            "/agents/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={},
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Unauthorized to access 'test'"}
