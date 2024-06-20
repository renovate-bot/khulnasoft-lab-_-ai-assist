from typing import Any, List, Optional, Type
from unittest import mock

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.agents import Agent
from ai_gateway.agents.chat.react import ReActAgent
from ai_gateway.api.v1 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.gitlab_features import GitLabUnitPrimitive


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


def mock_agent(
    prompt: str, expected_message: str, response: str, klass: Type[Agent] = Agent
) -> Agent:
    model = FakeModel(expected_message=expected_message, response=response)
    return klass(
        name="fake_agent",
        chain=ChatPromptTemplate.from_messages([prompt]) | model,
    )


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
    @mock.patch("ai_gateway.agents.registry.LocalAgentRegistry.get")
    @pytest.mark.parametrize(
        ("klass", "inputs", "expected_status", "expected_response"),
        [
            (Agent, {"name": "John", "age": 20}, 200, "Hi John!"),
            (
                None,
                {"name": "John", "age": 20},
                404,
                {"detail": "Agent 'test' not found for explain_vulnerability"},
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
        mock_registry_get,
        mock_container,
        mock_client,
        klass: Optional[Type[Agent]],
        inputs: dict[str, str],
        expected_status: int,
        expected_response: Any,
    ):
        if klass:
            mock_registry_get.return_value = mock_agent(
                prompt="Hi, I'm {name} and I'm {age} years old",
                expected_message="Hi, I'm John and I'm 20 years old",
                response="Hi John!",
                klass=klass,
            )
        else:
            mock_registry_get.side_effect = KeyError("test")

        response = mock_client.post(
            "/agents/invoke",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
            },
            json={"name": "test", "inputs": inputs},
        )

        mock_registry_get.assert_called_with(
            GitLabUnitPrimitive.EXPLAIN_VULNERABILITY, "test"
        )
        assert response.status_code == expected_status
        assert response.json() == expected_response


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(self, mock_container, mock_client):
        response = mock_client.post(
            "/agents/invoke",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Unit-Primitive": GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
            },
            json={"name": "test", "inputs": {}},
        )

        assert response.status_code == 403
        assert response.json() == {
            "detail": "Unauthorized to access explain_vulnerability"
        }
