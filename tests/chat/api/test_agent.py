from typing import AsyncIterator, Type
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from ai_gateway.api.v1.api import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.deps import ChatContainer
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicModel,
    ModelAPIError,
    SafetyAttributes,
    TextGenModelChunk,
    TextGenModelOutput,
)


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(scopes=["duo_chat"]),
    )


class TestAgentSuccessfulRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_body", "expected_provider_args"),
        [
            (
                {
                    "prompt_components": [
                        {
                            "type": "prompt",
                            "metadata": {
                                "source": "gitlab-rails-sm",
                                "version": "16.5.0-ee",
                            },
                            "payload": {
                                "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
                                "provider": "anthropic",
                                "model": "claude-2.0",
                            },
                        },
                    ]
                },
                {
                    "model_name": "claude-2.0",
                },
            ),
            (
                {
                    "prompt_components": [
                        {
                            "type": "prompt",
                            "metadata": {
                                "source": "gitlab-rails-sm",
                                "version": "16.5.0-ee",
                            },
                            "payload": {
                                "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
                                "provider": "anthropic",
                                "model": "claude-2.0",
                                "params": {
                                    "temperature": 0.3,
                                    "stop_sequences": ["\n\nHuman", "Observation:"],
                                    "max_tokens_to_sample": 1024,
                                },
                            },
                        },
                    ]
                },
                {
                    "model_name": "claude-2.0",
                    "temperature": 0.3,
                    "stop_sequences": ["\n\nHuman", "Observation:"],
                    "max_tokens_to_sample": 1024,
                },
            ),
        ],
    )
    async def test_successful_response(
        self, mock_client: TestClient, request_body: dict, expected_provider_args: dict
    ):
        mock_model = mock.Mock(spec=AnthropicModel)
        mock_model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="test completion",
                score=10000,
                safety_attributes=SafetyAttributes(),
            )
        )
        mock_anthropic_model = mock.Mock()
        mock_anthropic_model.provider.return_value = mock_model

        container = ChatContainer()

        with container.anthropic_model.override(mock_anthropic_model):
            response = mock_client.post(
                "/v1/chat/agent",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json=request_body,
            )

        assert response.status_code == 200

        assert response.json()["response"] == "test completion"

        response_metadata = response.json()["metadata"]
        assert response_metadata["provider"] == "anthropic"
        assert (
            response_metadata["model"]
            == request_body["prompt_components"][0]["payload"]["model"]
        )

        mock_anthropic_model.provider.assert_called_with(**expected_provider_args)
        mock_model.generate.assert_called_with(
            prefix="\n\nHuman: hello, what is your name?\n\nAssistant:",
            _suffix="",
            stream=False,
        )


class TestAgentSuccessfuStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_chunks", "expected_response"),
        [
            (
                [
                    TextGenModelChunk(
                        text="test",
                    ),
                    TextGenModelChunk(
                        text=" ",
                    ),
                    TextGenModelChunk(
                        text="completion",
                    ),
                ],
                "test completion",
            ),
        ],
    )
    async def test_successful_stream(
        self,
        mock_client: TestClient,
        model_chunks: list[TextGenModelChunk],
        expected_response: str,
    ):
        async def _stream_generator(
            prefix, _suffix, stream
        ) -> AsyncIterator[TextGenModelChunk]:
            for chunk in model_chunks:
                yield chunk

        model_name = "claude-2.0"
        mock_model = mock.Mock(spec=AnthropicModel)
        mock_model.generate = AsyncMock(side_effect=_stream_generator)
        mock_anthropic_model = mock.Mock()
        mock_anthropic_model.provider.return_value = mock_model

        container = ChatContainer()

        with container.anthropic_model.override(mock_anthropic_model):
            response = mock_client.post(
                "/v1/chat/agent",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
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
                                "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
                                "provider": "anthropic",
                                "model": model_name,
                            },
                        },
                    ],
                    "stream": "True",
                },
            )

        assert response.status_code == 200
        assert response.text == expected_response
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        mock_anthropic_model.provider.assert_called_with(model_name=model_name)
        mock_model.generate.assert_called_with(
            prefix="\n\nHuman: hello, what is your name?\n\nAssistant:",
            _suffix="",
            stream=True,
        )


class TestAgentUnsupportedProvider:
    def test_invalid_request(
        self,
        mock_client: TestClient,
    ):
        response = mock_client.post(
            "/v1/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
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
                            "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
                            "provider": "UNSUPPORTED_PROVIDER",
                            "model": "claude-2.0",
                        },
                    },
                ]
            },
        )

        assert response.status_code == 422


class TestAgentUnsupportedModel:
    def test_invalid_request(
        self,
        mock_client: TestClient,
    ):
        response = mock_client.post(
            "/v1/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
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
                            "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
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
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_invalid_scope(
        self,
        mock_client: TestClient,
    ):
        response = mock_client.post(
            "/v1/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
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
                            "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
                            "provider": "anthropic",
                            "model": "claude-2.0",
                        },
                    }
                ]
            },
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Forbidden"}


class TestAgentInvalidRequestMissingFields:
    def test_invalid_request_missing_fields(
        self,
        mock_client: TestClient,
    ):
        response = mock_client.post(
            "/v1/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
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
                    "url": "https://errors.pydantic.dev/2.5/v/missing",
                },
                {
                    "type": "missing",
                    "loc": ["body", "prompt_components", 0, "payload", "content"],
                    "msg": "Field required",
                    "input": {"provider": "anthropic", "model": "claude-2.0"},
                    "url": "https://errors.pydantic.dev/2.5/v/missing",
                },
            ]
        }


class TestAgentInvalidRequestManyPromptComponents:
    def test_invalid_request_many_prompt_components(
        self,
        mock_client: TestClient,
    ):
        response = mock_client.post(
            "/v1/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
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
                            "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
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
                                "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
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
                    "url": "https://errors.pydantic.dev/2.5/v/too_long",
                }
            ]
        }


class TestAgentUnsuccessfulAnthropicRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_exception_type", [AnthropicAPIStatusError, AnthropicAPIConnectionError]
    )
    async def test_fail_receiving_anthropic_response(
        self, mock_client: TestClient, model_exception_type: Type[ModelAPIError]
    ):
        def _side_effect(*_args, **_kwargs):
            raise exception

        if issubclass(model_exception_type, AnthropicAPIStatusError):
            model_exception_type.code = 404
        exception = model_exception_type("exception message")

        mock_model = mock.Mock(spec=AnthropicModel)
        mock_model.generate = AsyncMock(
            side_effect=_side_effect,
            return_value=TextGenModelOutput(
                text="test completion",
                score=10000,
                safety_attributes=SafetyAttributes(),
            ),
        )
        mock_anthropic_model = mock.Mock()
        mock_anthropic_model.provider.return_value = mock_model

        container = ChatContainer()

        with container.anthropic_model.override(mock_anthropic_model):
            with capture_logs() as cap_logs:
                response = mock_client.post(
                    "/v1/chat/agent",
                    headers={
                        "Authorization": "Bearer 12345",
                        "X-Gitlab-Authentication-Type": "oidc",
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
                                    "content": "\n\nHuman: hello, what is your name?\n\nAssistant:",
                                    "provider": "anthropic",
                                    "model": "claude-2.0",
                                },
                            }
                        ]
                    },
                )

        assert response.status_code == 200
        assert response.json()["response"] == ""

        assert cap_logs[0]["event"].startswith("failed to execute Anthropic request")
        assert cap_logs[0]["log_level"] == "error"
