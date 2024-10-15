from unittest.mock import Mock

import pytest
from dependency_injector import containers
from fastapi.testclient import TestClient

from ai_gateway.api.v3 import api_router
from ai_gateway.cloud_connector import User, UserClaims
from ai_gateway.tracking import SnowplowEventContext


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(
            scopes=["code_suggestions"], subject="1234", gitlab_realm="self-managed"
        ),
    )


class TestEditorContentCompletion:
    @pytest.mark.parametrize(
        ("mock_completions_legacy_output_texts", "expected_response"),
        [
            # non-empty suggestions from model
            (
                ["def search", "println"],
                {
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        },
                        {
                            "text": "println",
                            "index": 0,
                            "finish_reason": "length",
                        },
                    ],
                    "metadata": {
                        "model": {
                            "engine": "vertex-ai",
                            "name": "code-gecko",
                            "lang": "python",
                        },
                    },
                },
            ),
            # empty suggestions from model
            (
                [""],
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "vertex-ai",
                            "name": "code-gecko",
                            "lang": "python",
                        },
                    },
                },
            ),
        ],
    )
    def test_successful_response(
        self,
        mock_client: TestClient,
        mock_completions_legacy: Mock,
        mock_completions_legacy_output_texts: str,
        expected_response: dict,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "choices_count": 3,
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            "/code/completions",
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

        body = response.json()

        assert body["choices"] == expected_response["choices"]

        assert body["metadata"]["model"] == expected_response["metadata"]["model"]

        assert body["metadata"]["timestamp"] > 0

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
        )
        mock_completions_legacy.assert_called_with(
            prefix=payload["content_above_cursor"],
            suffix=payload["content_below_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            stream=False,
            code_context=None,
            candidate_count=3,
            snowplow_event_context=expected_snowplow_event,
        )

    @pytest.mark.parametrize(
        ("model_provider", "expected_code", "expected_response", "expected_model"),
        [
            (
                "vertex-ai",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "vertex-ai",
                    "name": "code-gecko@002",
                    "lang": "python",
                },
            ),
            (
                "anthropic",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "anthropic",
                    "name": "claude-instant-1.2",
                    "lang": "python",
                },
            ),
            # default provider
            (
                "",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "vertex-ai",
                    "name": "code-gecko@002",
                    "lang": "python",
                },
            ),
            # unknown provider
            (
                "some-provider",
                422,
                "",
                {},
            ),
        ],
    )
    def test_model_provider(
        self,
        mock_client: TestClient,
        mock_anthropic: Mock,
        mock_code_gecko: Mock,
        model_provider: str,
        expected_code: int,
        expected_response: str,
        expected_model: dict,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": model_provider or None,
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json=data,
        )

        assert response.status_code == expected_code

        if expected_code >= 400:
            # if we want 400+ status we don't need check the response
            return

        body = response.json()

        assert body["choices"] == expected_response["choices"]

        assert body["metadata"]["model"] == expected_model

        mock = mock_anthropic if model_provider == "anthropic" else mock_code_gecko

        mock.assert_called

    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_completions_stream: Mock,
        mock_suggestions_output_text: str,
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
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            "/code/completions",
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
        )
        mock_completions_stream.assert_called_with(
            prefix=payload["content_above_cursor"],
            suffix=payload["content_below_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            stream=True,
            code_context=None,
            snowplow_event_context=expected_snowplow_event,
        )


class TestEditorContentGeneration:
    @pytest.mark.parametrize(
        (
            "mock_suggestions_output_text",
            "mock_suggestions_model",
            "mock_suggestions_engine",
            "expected_response",
        ),
        [
            # non-empty suggestions from model
            (
                "def search",
                "code-gecko",
                "vertex-ai",
                {
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                    "metadata": {
                        "model": {
                            "engine": "vertex-ai",
                            "name": "code-gecko",
                            "lang": "python",
                        },
                    },
                },
            ),
            # empty suggestions from model
            (
                "",
                "code-gecko",
                "vertex-ai",
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "vertex-ai",
                            "name": "code-gecko",
                            "lang": "python",
                        },
                    },
                },
            ),
        ],
    )
    def test_successful_response(
        self,
        mock_client: TestClient,
        mock_generations: Mock,
        mock_suggestions_output_text: str,
        expected_response: dict,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
        }

        prompt_component = {
            "type": "code_editor_generation",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            "/code/completions",
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

        body = response.json()

        assert body["choices"] == expected_response["choices"]

        assert body["metadata"]["model"] == expected_response["metadata"]["model"]

        assert body["metadata"]["timestamp"] > 0

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
        )
        mock_generations.assert_called_with(
            prefix=payload["content_above_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            model_provider=None,
            stream=False,
            snowplow_event_context=expected_snowplow_event,
            prompt_enhancer=None,
        )

    @pytest.mark.parametrize(
        (
            "mock_suggestions_output_text",
            "mock_suggestions_model",
            "mock_suggestions_engine",
            "expected_response",
        ),
        [
            # non-empty suggestions from model
            (
                "def search",
                "Claude 3 Code Generations Agent",
                "agent",
                {
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                    "metadata": {
                        "model": {
                            "engine": "agent",
                            "name": "Claude 3 Code Generations Agent",
                            "lang": "python",
                        },
                    },
                },
            ),
            # empty suggestions from model
            (
                "",
                "Claude 3 Code Generations Agent",
                "agent",
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "agent",
                            "name": "Claude 3 Code Generations Agent",
                            "lang": "python",
                        },
                    },
                },
            ),
        ],
    )
    def test_generation_agent_model_successful_response(
        self,
        mock_client: TestClient,
        mock_generations: Mock,
        mock_suggestions_output_text: str,
        expected_response: dict,
    ):
        prompt_enhancer = {
            "examples_array": [
                {
                    "example": "class Project:\n  def __init__(self, name, public):\n    self.name = name\n    self.visibility = 'PUBLIC' if public\n\n    # is this project public?\n{{cursor}}\n\n    # print name of this project",
                    "response": "<new_code>def is_public(self):\n  return self.visibility == 'PUBLIC'",
                    "trigger_type": "comment",
                },
                {
                    "example": "def get_user(session):\n  # get the current user's name from the session data\n{{cursor}}\n\n# is the current user an admin",
                    "response": "<new_code>username = None\nif 'username' in session:\n  username = session['username']\nreturn username",
                    "trigger_type": "comment",
                },
            ],
            "trimmed_prefix": "# Create a fast binary search\n",
            "trimmed_suffix": "",
            "related_files": [
                '<file_content file_name="main.go">\npackage main\n\nfunc main()\nfullName("John", "Doe")\n}\n\n</file_content>\n'
            ],
            "related_snippets": [
                '<snippet_content name="fullName">\nfunc fullName(first, last string) {\n  fmt.Println(first, last)\n}\n\n</snippet_content>\n'
            ],
            "libraries": ["some_library", "other_library"],
            "user_instruction": "Generate the best possible code based on instructions.",
        }

        data = {
            "prompt_components": [
                {
                    "type": "code_editor_generation",
                    "payload": {
                        "file_name": "main.py",
                        "content_above_cursor": "# Create a fast binary search\n",
                        "content_below_cursor": "\n",
                        "language_identifier": "python",
                        "prompt_id": "code_suggestions/generations",
                        "prompt_enhancer": prompt_enhancer,
                    },
                }
            ],
        }

        response = mock_client.post(
            "/code/completions",
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

        body = response.json()

        assert body["choices"] == expected_response["choices"]

        assert body["metadata"]["model"] == expected_response["metadata"]["model"]

        assert body["metadata"]["timestamp"] > 0

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
        )
        mock_generations.assert_called_with(
            prefix="# Create a fast binary search\n",
            file_name="main.py",
            editor_lang="python",
            model_provider=None,
            stream=False,
            snowplow_event_context=expected_snowplow_event,
            prompt_enhancer=prompt_enhancer,
        )

    @pytest.mark.parametrize(
        ("prompt", "want_called"),
        [
            # non-empty suggestions from model
            (
                "",
                False,
            ),
            # empty suggestions from model
            (
                "some prompt",
                True,
            ),
            (
                None,
                False,
            ),
        ],
    )
    def test_prompt(
        self,
        mock_client,
        mock_container: containers.Container,
        mock_generations: Mock,
        mock_with_prompt_prepared: Mock,
        prompt: str,
        want_called: bool,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "prompt": prompt,
        }

        prompt_component = {
            "type": "code_editor_generation",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json=data,
        )

        assert response.status_code == 200

        assert mock_with_prompt_prepared.called == want_called
        if want_called:
            mock_with_prompt_prepared.assert_called_with(prompt)

    @pytest.mark.parametrize(
        (
            "model_provider",
            "expected_code",
            "expected_response",
            "expected_model",
        ),
        [
            (
                "vertex-ai",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "vertex-ai",
                    "name": "code-bison@002",
                    "lang": "python",
                },
            ),
            (
                "anthropic",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "anthropic",
                    "name": "claude-2.0",
                    "lang": "python",
                },
            ),
            # default provider
            (
                "",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "vertex-ai",
                    "name": "code-bison@002",
                    "lang": "python",
                },
            ),
            # unknown provider
            (
                "some-provider",
                422,
                "",
                {},
            ),
        ],
    )
    def test_model_provider(
        self,
        mock_client: TestClient,
        mock_anthropic: Mock,
        mock_code_bison: Mock,
        model_provider: str,
        expected_code: int,
        expected_response: str,
        expected_model: dict,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": model_provider or None,
        }

        prompt_component = {
            "type": "code_editor_generation",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json=data,
        )

        assert response.status_code == expected_code

        if expected_code >= 400:
            # if we want 400+ status we don't need check the response
            return

        body = response.json()

        assert body["choices"] == expected_response["choices"]
        assert body["metadata"]["model"] == expected_model

    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_generations_stream: Mock,
        mock_suggestions_output_text: str,
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
            "/code/completions",
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


class TestUnauthorizedScopes:
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

    def test_failed_authorization_scope(self, mock_client):
        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "code_editor_completion",
                        "payload": {
                            "file_name": "test",
                            "content_above_cursor": "def hello_world():",
                            "content_below_cursor": "",
                            "model_provider": "vertex-ai",
                        },
                    }
                ]
            },
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Unauthorized to access code suggestions"}


class TestIncomingRequest:
    @pytest.mark.parametrize(
        ("request_body", "expected_code"),
        [
            # valid request
            (
                {
                    "prompt_components": [
                        {
                            "type": "code_editor_completion",
                            "payload": {
                                "file_name": "test",
                                "content_above_cursor": "def hello_world():",
                                "content_below_cursor": "",
                                # FIXME: Forcing anthropic as vertex-ai is not working
                                "model_provider": "anthropic",
                            },
                        },
                    ],
                },
                200,
            ),
            # unknown component type
            (
                {
                    "prompt_components": [
                        {
                            "type": "some_type",
                            "payload": {
                                "file_name": "test",
                                "content_above_cursor": "def hello_world():",
                                "content_below_cursor": "",
                            },
                        },
                    ],
                },
                422,
            ),
            # too many prompt_components
            (
                {
                    "prompt_components": [
                        {
                            "type": "code_editor_completion",
                            "payload": {
                                "file_name": "test",
                                "content_above_cursor": "def hello_world():",
                                "content_below_cursor": "",
                            },
                        },
                    ]
                    * 101,
                },
                422,
            ),
            # missing required field
            (
                {
                    "prompt_components": [
                        {
                            "type": "code_editor_completion",
                            "payload": {
                                "content_above_cursor": "def hello_world():",
                                "content_below_cursor": "",
                            },
                        },
                    ],
                },
                422,
            ),
        ],
    )
    def test_valid_request(
        self,
        mock_client: TestClient,
        mock_completions: Mock,
        request_body: dict,
        expected_code: int,
    ):
        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json=request_body,
        )

        assert response.status_code == expected_code


class TestUnauthorizedIssuer:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(
                scopes=["code_suggestions"],
                subject="1234",
                gitlab_realm="self-managed",
                issuer="gitlab-ai-gateway",
            ),
        )

    def test_failed_authorization_scope(self, mock_client):
        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "code_editor_completion",
                        "payload": {
                            "file_name": "test",
                            "content_above_cursor": "def hello_world():",
                            "content_below_cursor": "",
                            "model_provider": "vertex-ai",
                        },
                    }
                ]
            },
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Unauthorized to access code suggestions"}
