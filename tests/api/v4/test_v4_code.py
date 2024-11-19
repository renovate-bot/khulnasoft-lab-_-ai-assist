from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ai_gateway.api.v4 import api_router
from ai_gateway.tracking import SnowplowEventContext

# Pytest runs these imported tests as part of this file.
from tests.api.v3.test_v3_code import (  # pylint: disable=unused-import
    TestEditorContentCompletion,
    TestEditorContentGeneration,
    TestIncomingRequest,
    TestUnauthorizedIssuer,
    TestUnauthorizedScopes,
    auth_user,
)


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

        assert response.status_code == 200
        assert response.text == mock_suggestions_output_text
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        expected_snowplow_event = SnowplowEventContext(
            prefix_length=30,
            suffix_length=1,
            language="python",
            gitlab_realm="self-managed",
            is_direct_connection=False,
            gitlab_instance_id="1234",
            gitlab_global_user_id="test-user-id",
            gitlab_host_name="",
            gitlab_saas_duo_pro_namespace_ids=[],
            suggestion_source="network",
            region="us-central1",
        )
        mock_completions_stream.assert_called_with(
            prefix=payload["content_above_cursor"],
            suffix=payload["content_below_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            stream=True,
            code_context=None,
            snowplow_event_context=expected_snowplow_event,
            raw_prompt=None,
        )


class TestEditorContentGenerationStream:
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_generations_stream: Mock,
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
            },
            json=data,
        )

        assert response.status_code == 200
        assert response.text == mock_suggestions_output_text
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
