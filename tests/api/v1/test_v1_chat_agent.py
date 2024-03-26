from typing import Any, AsyncIterator, Type
from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from ai_gateway.api.v1 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.container import ContainerApplication
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    AnthropicAPITimeoutError,
    AnthropicChatModel,
    AnthropicModel,
    KindAnthropicModel,
    Message,
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


@pytest.fixture()
def mock_models():
    model_output = TextGenModelOutput(
        text="test completion",
        score=10000,
        safety_attributes=SafetyAttributes(),
    )
    mock_llm_model = mock.Mock(spec=AnthropicModel)
    mock_llm_model.generate = AsyncMock(return_value=model_output)

    mock_chat_model = mock.Mock(spec=AnthropicChatModel)
    mock_chat_model.generate = AsyncMock(return_value=model_output)

    container = ContainerApplication()
    with (
        container.chat._anthropic_claude_llm_factory.override(mock_llm_model),
        container.chat._anthropic_claude_chat_factory.override(mock_chat_model),
    ):
        yield {"llm": mock_llm_model, "chat": mock_chat_model}


@pytest.fixture()
def mock_models_stream():
    async def _stream(*args: Any, **kwargs: Any) -> AsyncIterator[TextGenModelChunk]:
        for chunk in ["test", " ", "completion"]:
            yield TextGenModelChunk(text=chunk)

    mock_llm_model = mock.Mock(spec=AnthropicModel)
    mock_llm_model.generate = AsyncMock(side_effect=_stream)

    mock_chat_model = mock.Mock(spec=AnthropicChatModel)
    mock_chat_model.generate = AsyncMock(side_effect=_stream)

    container = ContainerApplication()
    with (
        container.chat._anthropic_claude_llm_factory.override(mock_llm_model),
        container.chat._anthropic_claude_chat_factory.override(mock_chat_model),
    ):
        yield {"llm": mock_llm_model, "chat": mock_chat_model}


class TestAgentSuccessfulRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_body",
        [
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
                            "model": "claude-2.1",
                            "params": {
                                "temperature": 0.3,
                                "stop_sequences": ["\n\nHuman", "Observation:"],
                                "max_tokens_to_sample": 2048,
                            },
                        },
                    },
                ]
            },
            {
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": [
                                {
                                    "role": "system",
                                    "content": "You are a Python engineer",
                                },
                                {
                                    "role": "user",
                                    "content": "define a function that adds numbers together",
                                },
                            ],
                            "provider": "anthropic",
                            "model": "claude-3-opus-20240229",
                            "params": {
                                "temperature": 0.3,
                                "stop_sequences": ["\n\nHuman", "Observation:"],
                                "max_tokens_to_sample": 2048,
                            },
                        },
                    },
                ]
            },
            {
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": [
                                {
                                    "role": "system",
                                    "content": "You are a Python engineer",
                                },
                                {
                                    "role": "user",
                                    "content": "define a function that adds numbers together",
                                },
                            ],
                            "provider": "anthropic",
                            "model": "claude-3-sonnet-20240229",
                            "params": {
                                "temperature": 0.3,
                                "stop_sequences": ["\n\nHuman", "Observation:"],
                                "max_tokens_to_sample": 2048,
                            },
                        },
                    },
                ]
            },
            {
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": [
                                {
                                    "role": "system",
                                    "content": "You are a Python engineer",
                                },
                                {
                                    "role": "user",
                                    "content": "define a function that adds numbers together",
                                },
                            ],
                            "provider": "anthropic",
                            "model": "claude-3-haiku-20240307",
                            "params": {
                                "temperature": 0.3,
                                "stop_sequences": ["\n\nHuman", "Observation:"],
                                "max_tokens_to_sample": 2048,
                            },
                        },
                    },
                ]
            },
            {
                "prompt_components": [
                    {
                        "type": "prompt",
                        "metadata": {
                            "source": "gitlab-rails-sm",
                            "version": "16.5.0-ee",
                        },
                        "payload": {
                            "content": [
                                {
                                    "role": "system",
                                    "content": "You are a Python engineer",
                                },
                                {
                                    "role": "user",
                                    "content": "define a function that adds numbers together",
                                },
                            ],
                            "provider": "anthropic",
                            "model": "claude-3-haiku-20240307",
                        },
                    },
                ]
            },
        ],
    )
    async def test_successful_response(
        self,
        mock_client: TestClient,
        mock_models: dict,
        request_body: dict,
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json=request_body,
        )

        assert response.status_code == 200
        assert response.json()["response"] == "test completion"

        response_metadata = response.json()["metadata"]
        prompt_payload = request_body["prompt_components"][0]["payload"]
        prompt_params = prompt_payload.get("params", {})

        assert response_metadata["provider"] == "anthropic"
        assert response_metadata["model"] == prompt_payload["model"]

        if isinstance(prompt_payload["content"], str):
            mock_models["llm"].generate.assert_called_with(
                prefix=prompt_payload["content"], stream=False, **prompt_params
            )
        else:
            messages = [Message(**message) for message in prompt_payload["content"]]
            if max_tokens := prompt_params.pop("max_tokens_to_sample", None):
                prompt_params["max_tokens"] = max_tokens

            mock_models["chat"].generate.assert_called_with(
                messages=messages, stream=False, **prompt_params
            )


class TestAgentSuccessfulStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload_content",
        [
            "\n\nHuman: hello, what is your name?\n\nAssistant:",
            [{"role": "user", "content": "hello, what is your name?"}],
        ],
    )
    async def test_successful_stream(
        self,
        mock_client: TestClient,
        mock_models_stream: dict,
        payload_content: str | list[dict],
    ):
        response = mock_client.post(
            "/chat/agent",
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
                            "content": payload_content,
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

        if isinstance(payload_content, str):
            mock_models_stream["llm"].generate.assert_called_with(
                prefix=payload_content,
                stream=True,
            )
        else:
            messages = [Message(**content) for content in payload_content]
            mock_models_stream["chat"].generate.assert_called_with(
                messages=messages,
                stream=True,
            )


class TestAgentUnsupportedProvider:
    def test_invalid_request(
        self,
        mock_client: TestClient,
        mock_models: dict,
    ):
        response = mock_client.post(
            "/chat/agent",
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
    def test_invalid_request(self, mock_client: TestClient, mock_models: dict):
        response = mock_client.post(
            "/chat/agent",
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
        mock_models: dict,
    ):
        response = mock_client.post(
            "/chat/agent",
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
        mock_models: dict,
    ):
        response = mock_client.post(
            "/chat/agent",
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
                    "url": "https://errors.pydantic.dev/2.6/v/missing",
                },
                {
                    "type": "missing",
                    "loc": ["body", "prompt_components", 0, "payload", "content"],
                    "msg": "Field required",
                    "input": {"provider": "anthropic", "model": "claude-2.0"},
                    "url": "https://errors.pydantic.dev/2.6/v/missing",
                },
            ]
        }


class TestAgentInvalidRequestManyPromptComponents:
    def test_invalid_request_many_prompt_components(
        self,
        mock_client: TestClient,
        mock_models: dict,
    ):
        response = mock_client.post(
            "/chat/agent",
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
                    "url": "https://errors.pydantic.dev/2.6/v/too_long",
                }
            ]
        }


class TestAgentUnsuccessfulAnthropicRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_class", "model_exception_type"),
        [
            (AnthropicModel, AnthropicAPIStatusError),
            (AnthropicModel, AnthropicAPITimeoutError),
            (AnthropicModel, AnthropicAPIConnectionError),
            (AnthropicChatModel, AnthropicAPIStatusError),
            (AnthropicChatModel, AnthropicAPITimeoutError),
            (AnthropicChatModel, AnthropicAPIConnectionError),
        ],
    )
    async def test_fail_receiving_anthropic_response(
        self,
        mock_client: TestClient,
        model_class: Type[AnthropicModel | AnthropicChatModel],
        model_exception_type: Type[ModelAPIError],
    ):
        def _side_effect(*_args, **_kwargs):
            raise exception

        if issubclass(model_exception_type, AnthropicAPIStatusError):
            model_exception_type.code = 404
        exception = model_exception_type("exception message")

        mock_model = mock.Mock(spec=model_class)
        mock_model.generate = AsyncMock(side_effect=_side_effect)

        container = ContainerApplication()
        with (
            # override both models at the same time to avoid unnecessary if-else constructions
            container.chat._anthropic_claude_llm_factory.override(mock_model),
            container.chat._anthropic_claude_chat_factory.override(mock_model),
            patch("ai_gateway.api.v1.chat.agent.log_exception") as mock_log_exception,
            capture_logs(),
        ):
            response = mock_client.post(
                "/chat/agent",
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
