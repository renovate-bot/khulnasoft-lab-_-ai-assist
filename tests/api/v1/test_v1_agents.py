from typing import Any, List, Optional, Type
from unittest.mock import patch

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

from ai_gateway.agents import Agent
from ai_gateway.agents.typing import ModelMetadata
from ai_gateway.api.v1 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.chat.agents import ReActAgent
from ai_gateway.config import Config
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


@pytest.fixture
def mock_config():
    config = Config()
    config.custom_models.enabled = False

    yield config


@pytest.fixture
def model_factory():
    yield lambda model, **kwargs: FakeModel(
        expected_message="Hi, I'm John and I'm 20 years old",
        response="Hi John!",
    )


@pytest.fixture
def prompt_template():
    yield {"system": "Hi, I'm {name} and I'm {age} years old"}


@pytest.fixture
def mock_registry_get(request, agent_class: Optional[Type[Agent]]):
    with patch("ai_gateway.agents.registry.LocalAgentRegistry.get") as mock:
        if agent_class:
            mock.return_value = request.getfixturevalue("agent")
        else:
            mock.side_effect = KeyError()

        yield mock


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def unit_primitives():
    yield ["explain_vulnerability"]


@pytest.fixture
def auth_user(unit_primitives: list[GitLabUnitPrimitive]):
    return User(authenticated=True, claims=UserClaims(scopes=unit_primitives))


class TestAgent:
    @pytest.mark.parametrize(
        (
            "agent_class",
            "inputs",
            "model_metadata",
            "expected_status",
            "expected_response",
        ),
        [
            (Agent, {"name": "John", "age": 20}, None, 200, "Hi John!"),
            (
                Agent,
                {"name": "John", "age": 20},
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint="http://localhost:4000",
                    api_key="token",
                ),
                200,
                "Hi John!",
            ),
            (
                None,
                {"name": "John", "age": 20},
                None,
                404,
                {"detail": "Agent 'test' not found"},
            ),
            (
                Agent,
                {"name": "John"},
                None,
                422,
                {
                    "detail": "\"Input to ChatPromptTemplate is missing variables {'age'}.  Expected: ['age', 'name'] Received: ['name']\""
                },
            ),
            (
                ReActAgent,
                {"name": "John", "age": 20},
                None,
                422,
                {"detail": "Agent 'test' is not supported"},
            ),
        ],
    )
    def test_request(
        self,
        agent_class,
        mock_registry_get,
        mock_client,
        mock_track_internal_event,
        inputs: dict[str, str],
        model_metadata: Optional[ModelMetadata],
        expected_status: int,
        expected_response: Any,
    ):
        response = mock_client.post(
            "/agents/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "inputs": inputs,
                "model_metadata": model_metadata
                and model_metadata.model_dump(mode="json"),
            },
        )

        mock_registry_get.assert_called_with("test", None, model_metadata)
        assert response.status_code == expected_status
        assert response.json() == expected_response

        if agent_class:
            mock_track_internal_event.assert_called_once_with(
                "request_explain_vulnerability",
                category="ai_gateway.api.v1.agents.invoke",
            )
        else:
            mock_track_internal_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_streaming_request(
        self,
        mock_client,
        mock_registry_get,
    ):
        response = mock_client.post(
            "/agents/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "inputs": {"name": "John", "age": 20},
                "stream": True,
            },
        )

        mock_registry_get.assert_called_with("test", None, None)
        assert response.status_code == 200
        assert response.text == "Hi John!"
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


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
            json={"inputs": {}},
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Unauthorized to access 'test'"}
