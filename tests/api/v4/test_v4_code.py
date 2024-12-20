import json
from typing import Optional
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from ai_gateway.api.v4 import api_router
from ai_gateway.api.v4.code.typing import StreamEvent

# Pytest runs these imported tests as part of this file.
from tests.api.v3.test_v3_code import (  # pylint: disable=unused-import
    TestEditorContentCompletion,
    TestEditorContentGeneration,
    TestIncomingRequest,
    TestUnauthorizedIssuer,
    TestUnauthorizedScopes,
    auth_user,
)


@pytest.fixture(autouse=True)
def reset_sse_starlette_appstatus_event():
    # To avoid RuntimeError "bound to a different event loop" during SSE streaming in parameterized tests
    AppStatus.should_exit_event = None


@pytest.fixture
def route():
    return "/code/suggestions"


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


class TestEditorContentCompletionStream:
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_completions_stream: Mock,
        mock_suggestions_output_text: str,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": "anthropic",
            "stream": True,
            "model_name": "claude-3-5-sonnet-20240620",
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Global-User-Id": "test-user-id",
            },
            json=data,
        )

        expected_model_metadata = {
            "engine": "anthropic",
            "name": "claude-3-5-sonnet-20240620",
        }

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["X-Streaming-Format"] == "sse"

        _assert_stream_sse_responses(
            response.text, mock_suggestions_output_text, expected_model_metadata
        )


class TestEditorContentGenerationStream:
    @pytest.mark.parametrize(
        (
            "prompt_id",
            "model_provider",
            "expected_model_metadata",
        ),
        [
            (
                None,
                "vertex-ai",
                {
                    "engine": "vertex-ai",
                    "name": "code-bison@002",
                },
            ),
            (
                None,
                "anthropic",
                {
                    "engine": "anthropic",
                    "name": "claude-2.0",
                },
            ),
            (
                "code_suggestions/generations",
                None,
                {
                    "engine": "agent",
                    "name": "Claude 3 Code Generations Agent",
                },
            ),
        ],
    )
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_generations_stream: Mock,
        mock_suggestions_output_text: str,
        route: str,
        prompt_id: Optional[str],
        model_provider: Optional[str],
        expected_model_metadata: dict,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": model_provider,
            "prompt_id": prompt_id,
            "stream": True,
        }

        prompt_component = {
            "type": "code_editor_generation",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Global-User-Id": "test-user-id",
            },
            json=data,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["X-Streaming-Format"] == "sse"

        _assert_stream_sse_responses(
            response.text, mock_suggestions_output_text, expected_model_metadata
        )


def _assert_stream_sse_responses(
    response_text: str,
    expected_suggestions_output_text: str,
    expected_model_metadata: dict,
):
    def _parse_sse_messages():
        parsed_list = []
        for message in response_text.strip().split("\r\n\r\n"):
            lines = message.splitlines()
            event = lines[0].removeprefix("event: ")
            data = json.loads(lines[1].removeprefix("data: "))
            parsed_list.append({"event": event, "data": data})
        return parsed_list

    sse_messages = _parse_sse_messages()
    start_message = sse_messages.pop(0)
    end_message = sse_messages.pop()

    assert start_message["event"] == StreamEvent.START
    assert start_message["data"]["metadata"]["model"] == expected_model_metadata
    assert start_message["data"]["metadata"]["timestamp"] > 0

    assert end_message["event"] == StreamEvent.END
    assert end_message["data"] is None

    # _mock_async_execute() yields one character at a time, so we expect
    # a content chunk message for each character of the output text.
    assert len(expected_suggestions_output_text) == len(sse_messages)

    for index, content_message in enumerate(sse_messages):
        assert content_message["event"] == StreamEvent.CONTENT_CHUNK
        assert content_message["data"]["choices"][0]["index"] == 0
        assert (
            content_message["data"]["choices"][0]["delta"]["content"]
            == expected_suggestions_output_text[index]
        )
