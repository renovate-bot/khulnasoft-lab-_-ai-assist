from typing import Any, Type
from unittest.mock import Mock, patch

import pydantic
import pytest
from dependency_injector import containers
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from ai_gateway.api.v1 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicAPITimeoutError,
    KindAnthropicModel,
    Message,
    ModelAPIError,
)
from ai_gateway.models.base_text import TextGenModelChunk, TextGenModelOutput


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(
            scopes=["duo_chat"], subject="1234", gitlab_realm="self-managed"
        ),
    )


@pytest.fixture
def text_content():
    return "\n\nHuman: hello, what is your name?\n\nAssistant:"


@pytest.fixture
def chat_content():
    return [
        {
            "role": "system",
            "content": "You are a Python engineer",
        },
        {
            "role": "user",
            "content": "define a function that adds numbers together",
        },
    ]


class TestAgentSuccessfulRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("content_fixture", "provider", "model", "params"),
        [
            ("text_content", "anthropic", "claude-2.0", None),
            (
                "text_content",
                "anthropic",
                "claude-2.1",
                {
                    "temperature": 0.3,
                    "stop_sequences": ["\n\nHuman", "Observation:"],
                    "max_tokens_to_sample": 2048,
                },
            ),
            (
                "chat_content",
                "anthropic",
                "claude-3-opus-20240229",
                {
                    "temperature": 0.3,
                    "stop_sequences": ["\n\nHuman", "Observation:"],
                    "max_tokens_to_sample": 2048,
                },
            ),
            (
                "chat_content",
                "anthropic",
                "claude-3-sonnet-20240229",
                {
                    "temperature": 0.3,
                    "stop_sequences": ["\n\nHuman", "Observation:"],
                    "max_tokens_to_sample": 2048,
                },
            ),
            (
                "chat_content",
                "anthropic",
                "claude-3-haiku-20240307",
                {
                    "temperature": 0.3,
                    "stop_sequences": ["\n\nHuman", "Observation:"],
                    "max_tokens_to_sample": 2048,
                },
            ),
            ("chat_content", "anthropic", "claude-3-haiku-20240307", None),
            ("chat_content", "litellm", "mistral", None),
        ],
    )
    async def test_successful_response(
        self,
        request,
        mock_client: TestClient,
        mock_anthropic: Mock,
        mock_anthropic_chat: Mock,
        mock_llm_chat: Mock,
        content_fixture: str,
        provider: str,
        model: str,
        params: dict[str, Any] | None,
    ):
        content = request.getfixturevalue(content_fixture)
        payload = {
            "content": content,
            "provider": provider,
            "model": model,
        }
        if params:
            payload["params"] = params
        else:
            params = {}

        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": payload,
                    },
                ]
            },
        )

        assert response.status_code == 200
        assert response.json()["response"] == "test completion"

        response_metadata = response.json()["metadata"]

        assert response_metadata["provider"] == provider
        assert response_metadata["model"] == model

        if isinstance(content, str):
            mock_anthropic.assert_called_with(prefix=content, stream=False, **params)
        else:
            messages = [Message(**message) for message in content]
            if max_tokens := params.pop("max_tokens_to_sample", None):
                params["max_tokens"] = max_tokens

            mock = mock_anthropic_chat if provider == "anthropic" else mock_llm_chat
            mock.assert_called_with(messages=messages, stream=False, **params)


class TestAgentSuccessfulStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("content_fixture", ["text_content", "chat_content"])
    async def test_successful_stream(
        self,
        request,
        mock_client: TestClient,
        mock_anthropic_stream: Mock,
        mock_anthropic_chat_stream: Mock,
        content_fixture: str,
    ):
        content = request.getfixturevalue(content_fixture)
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": content,
                            "provider": "anthropic",
                            "model": KindAnthropicModel.CLAUDE_2_0.value,
                        },
                    },
                ],
                "stream": "True",
            },
        )

        assert response.status_code == 200
        assert response.text == "test completion"
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        if isinstance(content, str):
            mock_anthropic_stream.assert_called_with(
                prefix=content,
                stream=True,
            )
        else:
            messages = [Message(**content) for content in content]
            mock_anthropic_chat_stream.assert_called_with(
                messages=messages,
                stream=True,
            )


class TestAgentUnsupportedProvider:
    def test_invalid_request(
        self,
        mock_client: TestClient,
        text_content: str,
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": text_content,
                            "provider": "UNSUPPORTED_PROVIDER",
                            "model": "claude-2.0",
                        },
                    },
                ]
            },
        )

        assert response.status_code == 422


class TestAgentUnsupportedModel:
    def test_invalid_request(self, mock_client: TestClient, text_content: str):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": text_content,
                            "provider": "anthropic",
                            "model": "UNSUPPORTED_MODEL",
                        },
                    },
                ]
            },
        )

        assert response.status_code == 422


class TestAnthropicInvalidScope:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(
                scopes=["unauthorized_scope"],
                subject="1234",
                gitlab_realm="self-managed",
            ),
        )

    def test_invalid_scope(
        self,
        mock_client: TestClient,
        text_content: str,
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": text_content,
                            "provider": "anthropic",
                            "model": "claude-2.0",
                        },
                    }
                ]
            },
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Unauthorized to access duo chat"}


class TestAgentInvalidRequestMissingFields:
    def test_invalid_request_missing_fields(
        self,
        mock_client: TestClient,
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {"source": "gitlab-rails-sm"},
                        "payload": {
                            "provider": "anthropic",
                            "model": "claude-2.0",
                        },
                    },
                ]
            },
        )

        assert response.status_code == 422
        assert response.json() == {
            "detail": [
                {
                    "type": "missing",
                    "loc": ["body", "prompt_components", 0, "metadata", "version"],
                    "msg": "Field required",
                    "input": {"source": "gitlab-rails-sm"},
                },
                {
                    "type": "missing",
                    "loc": ["body", "prompt_components", 0, "payload", "content"],
                    "msg": "Field required",
                    "input": {"provider": "anthropic", "model": "claude-2.0"},
                },
            ]
        }


class TestAgentInvalidRequestManyPromptComponents:
    def test_invalid_request_many_prompt_components(
        self,
        mock_client: TestClient,
        text_content: str,
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": text_content,
                            "provider": "anthropic",
                            "model": "claude-2.0",
                        },
                    },
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "vscode",
                            "version": "1.2.3",
                        },
                        "payload": {
                            "content": "SECOND PROMPT COMPONENT (NOT EXPECTED)",
                            "provider": "anthropic",
                            "model": "claude-2.0",
                        },
                    },
                ]
            },
        )

        assert response.status_code == 422
        assert response.json() == {
            "detail": [
                {
                    "type": "too_long",
                    "loc": ["body", "prompt_components"],
                    "msg": "List should have at most 1 item after validation, not 2",
                    "input": [
                        {
                            "type": "prompt",
                            "metadata": {
                                "source": "gitlab-rails-sm",
                                "version": "16.5.0-ee",
                            },
                            "payload": {
                                "content": text_content,
                                "provider": "anthropic",
                                "model": "claude-2.0",
                            },
                        },
                        {
                            "type": "prompt",
                            "metadata": {"source": "vscode", "version": "1.2.3"},
                            "payload": {
                                "content": "SECOND PROMPT COMPONENT (NOT EXPECTED)",
                                "provider": "anthropic",
                                "model": "claude-2.0",
                            },
                        },
                    ],
                    "ctx": {"field_type": "List", "max_length": 1, "actual_length": 2},
                }
            ]
        }


class TestAgentUnsuccessfulAnthropicRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("content_fixture", "model_exception_type"),
        [
            ("text_content", AnthropicAPIStatusError),
            ("text_content", AnthropicAPITimeoutError),
            ("text_content", AnthropicAPIConnectionError),
            ("chat_content", AnthropicAPIStatusError),
            ("chat_content", AnthropicAPITimeoutError),
            ("chat_content", AnthropicAPIConnectionError),
        ],
    )
    async def test_fail_receiving_anthropic_response(
        self,
        request,
        mock_client: TestClient,
        mock_anthropic: Mock,
        mock_anthropic_chat: Mock,
        content_fixture: str,
        model_exception_type: Type[ModelAPIError],
    ):
        def _side_effect(*_args, **_kwargs):
            raise exception

        if issubclass(model_exception_type, AnthropicAPIStatusError):
            model_exception_type.code = 404
        exception = model_exception_type("exception message")

        mock_anthropic.side_effect = _side_effect
        mock_anthropic_chat.side_effect = _side_effect

        with (
            patch("ai_gateway.api.v1.chat.agent.log_exception") as mock_log_exception,
            capture_logs(),
        ):
            response = mock_client.post(
                "/chat/agent",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                },
                json={
                    "prompt_components": [
                        {
                            "type": "prompt",
                            "metadata": {
                                "source": "gitlab-rails-sm",
                                "version": "16.5.0-ee",
                            },
                            "payload": {
                                "content": request.getfixturevalue(content_fixture),
                                "provider": "anthropic",
                                "model": "claude-2.0",
                            },
                        }
                    ]
                },
            )

            mock_log_exception.assert_called_once()

        if issubclass(model_exception_type, AnthropicAPIStatusError):
            assert response.status_code == 502
            assert response.json()["detail"] == "Anthropic API Status Error."
        elif issubclass(model_exception_type, AnthropicAPITimeoutError):
            assert response.status_code == 504
            assert response.json()["detail"] == "Anthropic API Timeout Error."
        else:
            assert response.status_code == 502
            assert response.json()["detail"] == "Anthropic API Connection Error."
