from typing import AsyncIterator
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from ai_gateway.api.v3 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeGenerations,
    CodeSuggestionsChunk,
    CodeSuggestionsOutput,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId
from ai_gateway.container import ContainerApplication
from ai_gateway.models import ModelMetadata


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
    def auth_user_default_issued():
        return User(
            authenticated=True,
            claims=UserClaims(
                scopes=["code_suggestions"], subject="1234", gitlab_realm="self-managed"
            ),
        )

    def auth_user_aigw_issued():
        return User(
            authenticated=True,
            claims=UserClaims(
                scopes=["code_suggestions"],
                subject="1234",
                gitlab_realm="self-managed",
                issuer="gitlab-ai-gateway",
            ),
        )

    @pytest.mark.parametrize(
        ("model_output", "expected_response"),
        [
            # non-empty suggestions from model
            (
                CodeSuggestionsOutput(
                    text="def search",
                    score=0,
                    model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "response": "def search",
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
                CodeSuggestionsOutput(
                    text="",
                    score=0,
                    model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "response": "",
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
        model_output: CodeSuggestionsOutput,
        expected_response: dict,
    ):
        code_completions_mock = mock.Mock(spec=CodeCompletions)
        code_completions_mock.execute = mock.AsyncMock(return_value=model_output)
        container = ContainerApplication()

        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        with container.code_suggestions.completions.vertex_legacy.override(
            code_completions_mock
        ):
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

        body = response.json()

        assert body["response"] == expected_response["response"]

        assert body["metadata"]["model"] == expected_response["metadata"]["model"]

        assert body["metadata"]["timestamp"] > 0

        code_completions_mock.execute.assert_called_with(
            prefix=payload["content_above_cursor"],
            suffix=payload["content_below_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            stream=False,
            code_context=None,
        )

    @pytest.mark.parametrize(
        ("model_provider", "expected_code", "expected_response", "expected_model"),
        [
            (
                "vertex-ai",
                200,
                "vertex response",
                {
                    "engine": "vertex-ai",
                    "name": "code-gecko",
                    "lang": "python",
                },
            ),
            (
                "anthropic",
                200,
                "anthropic response",
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
                "vertex response",
                {
                    "engine": "vertex-ai",
                    "name": "code-gecko",
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
        model_provider: str,
        expected_code: int,
        expected_response: str,
        expected_model: dict,
    ):
        vertex_output = CodeSuggestionsOutput(
            text="vertex response",
            score=0,
            model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
            lang_id=LanguageId.PYTHON,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
            ),
        )

        anthropic_output = CodeSuggestionsOutput(
            text="anthropic response",
            score=0,
            model=ModelMetadata(name="claude-instant-1.2", engine="anthropic"),
            lang_id=LanguageId.PYTHON,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
            ),
        )
        vertex_mock = mock.Mock(spec=CodeCompletions)
        vertex_mock.execute = mock.AsyncMock(return_value=vertex_output)

        anthropic_mock = mock.Mock(spec=CodeCompletions)
        anthropic_mock.execute = mock.AsyncMock(return_value=anthropic_output)
        container = ContainerApplication()

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

        with container.code_suggestions.completions.vertex_legacy.override(
            vertex_mock
        ) and container.code_suggestions.completions.anthropic.override(anthropic_mock):
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

        assert body["response"] == expected_response

        assert body["metadata"]["model"] == expected_model

    @pytest.mark.parametrize(
        ("model_chunks", "expected_response"),
        [
            (
                [
                    CodeSuggestionsChunk(
                        text="def search",
                    ),
                    CodeSuggestionsChunk(
                        text=" (query)",
                    ),
                ],
                "def search (query)",
            ),
        ],
    )
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        model_chunks: list[CodeSuggestionsChunk],
        expected_response: str,
    ):
        async def _stream_generator(
            prefix, suffix, file_name, editor_lang, stream, code_context
        ) -> AsyncIterator[CodeSuggestionsChunk]:
            for chunk in model_chunks:
                yield chunk

        code_completions_mock = mock.Mock(spec=CodeCompletions)
        code_completions_mock.execute = mock.AsyncMock(side_effect=_stream_generator)
        container = ContainerApplication()

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

        with container.code_suggestions.completions.anthropic.override(
            code_completions_mock
        ):
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
        assert response.text == expected_response
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        code_completions_mock.execute.assert_called_with(
            prefix=payload["content_above_cursor"],
            suffix=payload["content_below_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            stream=True,
            code_context=None,
        )

    @pytest.mark.parametrize(
        ("auth_user", "expected_status_code"),
        [
            (
                User(
                    authenticated=True,
                    claims=UserClaims(
                        scopes=["code_suggestions"],
                        subject="1234",
                        gitlab_realm="self-managed",
                    ),
                ),
                200,
            ),
            (
                User(
                    authenticated=True,
                    claims=UserClaims(
                        scopes=["code_suggestions"],
                        subject="1234",
                        gitlab_realm="self-managed",
                        issuer="gitlab-ai-gateway",
                    ),
                ),
                200,
            ),
        ],
    )
    def test_successful_response_with_correct_issuers(
        self, mock_client: TestClient, auth_user: User, expected_status_code: int
    ):
        code_completions_mock = mock.Mock(spec=CodeCompletions)
        code_completions_mock.execute = mock.AsyncMock(
            return_value=CodeSuggestionsOutput(
                text="def search",
                score=0,
                model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                lang_id=LanguageId.PYTHON,
                metadata=CodeSuggestionsOutput.Metadata(
                    experiments=[],
                ),
            )
        )
        container = ContainerApplication()

        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        with container.code_suggestions.completions.vertex_legacy.override(
            code_completions_mock
        ):
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

        assert response.status_code == expected_status_code


class TestEditorContentGeneration:
    @pytest.mark.parametrize(
        ("model_output", "expected_response"),
        [
            # non-empty suggestions from model
            (
                CodeSuggestionsOutput(
                    text="def search",
                    score=0,
                    model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "response": "def search",
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
                CodeSuggestionsOutput(
                    text="",
                    score=0,
                    model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "response": "",
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
        model_output: CodeSuggestionsOutput,
        expected_response: dict,
    ):
        code_generations_mock = mock.Mock(spec=CodeGenerations)
        code_generations_mock.execute = mock.AsyncMock(return_value=model_output)
        container = ContainerApplication()

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

        with container.code_suggestions.generations.vertex.override(
            code_generations_mock
        ):
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

        body = response.json()

        assert body["response"] == expected_response["response"]

        assert body["metadata"]["model"] == expected_response["metadata"]["model"]

        assert body["metadata"]["timestamp"] > 0

        code_generations_mock.execute.assert_called_with(
            prefix=payload["content_above_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            model_provider=None,
            stream=False,
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
        prompt: str,
        want_called: bool,
    ):
        model_output = CodeSuggestionsOutput(
            text="def search",
            score=0,
            model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
            lang_id=LanguageId.PYTHON,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
            ),
        )
        code_generations_mock = mock.Mock(spec=CodeGenerations)
        code_generations_mock.execute = mock.AsyncMock(return_value=model_output)
        code_generations_mock.with_prompt_prepared = mock.AsyncMock()
        container = ContainerApplication()

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

        with container.code_suggestions.generations.vertex.override(
            code_generations_mock
        ):
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

        assert code_generations_mock.with_prompt_prepared.called == want_called
        if want_called:
            code_generations_mock.with_prompt_prepared.assert_called_with(prompt)

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
                "vertex response",
                {
                    "engine": "vertex-ai",
                    "name": "code-gecko",
                    "lang": "python",
                },
            ),
            (
                "anthropic",
                200,
                "anthropic response",
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
                "vertex response",
                {
                    "engine": "vertex-ai",
                    "name": "code-gecko",
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
        model_provider: str,
        expected_code: int,
        expected_response: str,
        expected_model: dict,
    ):
        vertex_output = CodeSuggestionsOutput(
            text="vertex response",
            score=0,
            model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
            lang_id=LanguageId.PYTHON,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
            ),
        )

        anthropic_output = CodeSuggestionsOutput(
            text="anthropic response",
            score=0,
            model=ModelMetadata(name="claude-instant-1.2", engine="anthropic"),
            lang_id=LanguageId.PYTHON,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
            ),
        )
        vertex_mock = mock.Mock(spec=CodeGenerations)
        vertex_mock.execute = mock.AsyncMock(return_value=vertex_output)

        anthropic_mock = mock.Mock(spec=CodeGenerations)
        anthropic_mock.execute = mock.AsyncMock(return_value=anthropic_output)
        container = ContainerApplication()

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

        with container.code_suggestions.generations.vertex.override(
            vertex_mock
        ) and container.code_suggestions.generations.anthropic_default.override(
            anthropic_mock
        ):
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

        assert body["response"] == expected_response
        assert body["metadata"]["model"] == expected_model

    @pytest.mark.parametrize(
        ("model_chunks", "expected_response"),
        [
            (
                [
                    CodeSuggestionsChunk(
                        text="def search",
                    ),
                    CodeSuggestionsChunk(
                        text=" (query)",
                    ),
                ],
                "def search (query)",
            ),
        ],
    )
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        model_chunks: list[CodeSuggestionsChunk],
        expected_response: str,
    ):
        async def _stream_generator(
            prefix: str,
            file_name: str,
            editor_lang: str,
            model_provider: str,
            stream: bool,
        ) -> AsyncIterator[CodeSuggestionsChunk]:
            for chunk in model_chunks:
                yield chunk

        code_generations_mock = mock.Mock(spec=CodeGenerations)
        code_generations_mock.execute = mock.AsyncMock(side_effect=_stream_generator)
        container = ContainerApplication()

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

        with container.code_suggestions.generations.anthropic_default.override(
            code_generations_mock
        ):
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
        assert response.text == expected_response
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
        request_body: dict,
        expected_code: int,
    ):
        model_output = CodeSuggestionsOutput(
            text="vertex response",
            score=0,
            model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
            lang_id=LanguageId.PYTHON,
            metadata=CodeSuggestionsOutput.Metadata(
                experiments=[],
            ),
        )
        code_completions_mock = mock.Mock(spec=CodeCompletions)
        code_completions_mock.execute = mock.AsyncMock(return_value=model_output)
        container = ContainerApplication()

        with container.code_suggestions.completions.vertex_legacy.override(
            code_completions_mock
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
