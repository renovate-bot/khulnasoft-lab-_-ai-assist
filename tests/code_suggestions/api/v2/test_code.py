from typing import AsyncIterator
from unittest import mock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from snowplow_tracker import Snowplow

from ai_gateway.api.v2.api import api_router
from ai_gateway.api.v2.code.api import (
    CurrentFile,
    SuggestionsRequest,
    track_snowplow_event,
)
from ai_gateway.auth import User, UserClaims
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsChunk,
    CodeSuggestionsOutput,
)
from ai_gateway.code_suggestions.processing.base import ModelEngineOutput
from ai_gateway.code_suggestions.processing.typing import (
    LanguageId,
    MetadataCodeContent,
    MetadataPromptBuilder,
)
from ai_gateway.deps import CodeSuggestionsContainer
from ai_gateway.experimentation.base import ExperimentTelemetry
from ai_gateway.instrumentators.base import Telemetry
from ai_gateway.models import ModelMetadata
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(scopes=["code_suggestions"]),
    )


class TestCodeCompletions:
    @pytest.mark.parametrize(
        ("model_output", "expected_response"),
        [
            # non-empty suggestions from model
            (
                ModelEngineOutput(
                    text="def search",
                    score=0,
                    model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                    lang_id=LanguageId.PYTHON,
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=10, length_tokens=2),
                            "suffix": MetadataCodeContent(length=10, length_tokens=2),
                        },
                        experiments=[
                            ExperimentTelemetry(name="truncate_suffix", variant=1)
                        ],
                    ),
                ),
                {
                    "id": "id",
                    "model": {
                        "engine": "vertex-ai",
                        "name": "code-gecko",
                        "lang": "python",
                    },
                    "experiments": [{"name": "truncate_suffix", "variant": 1}],
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                },
            ),
            # empty suggestions from model
            (
                ModelEngineOutput(
                    text="",
                    score=0,
                    model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                    lang_id=LanguageId.PYTHON,
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=10, length_tokens=2),
                            "suffix": MetadataCodeContent(length=10, length_tokens=2),
                        },
                        experiments=[
                            ExperimentTelemetry(name="truncate_suffix", variant=1)
                        ],
                    ),
                ),
                {
                    "id": "id",
                    "model": {
                        "engine": "vertex-ai",
                        "name": "code-gecko",
                        "lang": "python",
                    },
                    "experiments": [{"name": "truncate_suffix", "variant": 1}],
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [],
                },
            ),
        ],
    )
    def test_legacy_successful_response(
        self,
        mock_client: TestClient,
        model_output: ModelEngineOutput,
        expected_response: dict,
    ):
        code_completions_mock = mock.Mock(spec=CodeCompletionsLegacy)
        code_completions_mock.execute = mock.AsyncMock(return_value=model_output)
        container = CodeSuggestionsContainer()

        with container.code_completions_legacy.override(code_completions_mock):
            response = mock_client.post(
                "/v2/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json={
                    "prompt_version": 1,
                    "project_path": "gitlab-org/gitlab",
                    "project_id": 278964,
                    "current_file": {
                        "file_name": "main.py",
                        "content_above_cursor": "# Create a fast binary search\n",
                        "content_below_cursor": "\n",
                    },
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["choices"] == expected_response["choices"]
        assert body["experiments"] == expected_response["experiments"]
        assert body["model"] == expected_response["model"]

    @pytest.mark.parametrize(
        ("prompt_version", "model_output", "expected_response"),
        [
            # non-empty suggestions from model
            (
                1,
                CodeSuggestionsOutput(
                    text="def search",
                    score=0,
                    model=ModelMetadata(name="claude-instant-1.2", engine="anthropic"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-instant-1.2",
                        "lang": "python",
                    },
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                },
            ),
            # prompt version 2
            (
                2,
                CodeSuggestionsOutput(
                    text="def search",
                    score=0,
                    model=ModelMetadata(name="claude-instant-1.2", engine="anthropic"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-instant-1.2",
                        "lang": "python",
                    },
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                },
            ),
            # empty suggestions from model
            (
                1,
                CodeSuggestionsOutput(
                    text="",
                    score=0,
                    model=ModelMetadata(name="claude-instant-1.2", engine="anthropic"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-instant-1.2",
                        "lang": "python",
                    },
                    "object": "text_completion",
                    "created": 1695182638,
                    "choices": [],
                },
            ),
        ],
    )
    def test_successful_response(
        self,
        prompt_version: int,
        mock_client: TestClient,
        model_output: CodeSuggestionsOutput,
        expected_response: dict,
    ):
        code_completions_mock = mock.Mock(spec=CodeCompletions)
        code_completions_mock.execute = mock.AsyncMock(return_value=model_output)
        container = CodeSuggestionsContainer()

        current_file = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
        }
        data = {
            "prompt_version": prompt_version,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "model_provider": "anthropic",
            "current_file": current_file,
        }

        code_completions_kwargs = {}
        if prompt_version == 2:
            data.update(
                {
                    "prompt": current_file["content_above_cursor"],
                }
            )
            code_completions_kwargs.update(
                {"raw_prompt": current_file["content_above_cursor"]}
            )

        with container.code_completions_anthropic.override(code_completions_mock):
            response = mock_client.post(
                "/v2/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json=data,
            )

        assert response.status_code == 200

        body = response.json()

        assert body["choices"] == expected_response["choices"]
        assert body["model"] == expected_response["model"]

        code_completions_mock.execute.assert_called_with(
            prefix=current_file["content_above_cursor"],
            suffix=current_file["content_below_cursor"],
            file_name=current_file["file_name"],
            editor_lang=current_file.get("language_identifier", None),
            stream=False,
            **code_completions_kwargs,
        )

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
            prefix, suffix, file_name, editor_lang, stream
        ) -> AsyncIterator[CodeSuggestionsChunk]:
            for chunk in model_chunks:
                yield chunk

        code_completions_mock = mock.Mock(spec=CodeCompletions)
        code_completions_mock.execute = mock.AsyncMock(side_effect=_stream_generator)
        container = CodeSuggestionsContainer()

        current_file = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
        }
        data = {
            "prompt_version": 1,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "model_provider": "anthropic",
            "current_file": current_file,
            "stream": True,
        }

        with container.code_completions_anthropic.override(code_completions_mock):
            response = mock_client.post(
                "/v2/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json=data,
            )

        assert response.status_code == 200
        assert response.text == expected_response
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        code_completions_mock.execute.assert_called_with(
            prefix=current_file["content_above_cursor"],
            suffix=current_file["content_below_cursor"],
            file_name=current_file["file_name"],
            editor_lang=current_file.get("language_identifier", None),
            stream=True,
        )


class TestSnowplowInstrumentator:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @pytest.mark.parametrize(
        (
            "request_headers",
            "jwt_realm_claim",
            "expected_instance_id",
            "expected_user_id",
            "expected_gitlab_host_name",
            "expected_gitlab_saas_namespace_ids",
            "expected_realm",
        ),
        [
            (
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Host-Name": "gitlab.com",
                    "X-Gitlab-Saas-Namespace-Ids": "1,2,3",
                    "X-Gitlab-Realm": "saas",
                },
                None,
                "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                "gitlab.com",
                ["1", "2", "3"],
                "saas",
            ),
            (
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Host-Name": "awesome-org.com",
                    "X-Gitlab-Realm": "self-managed",
                },
                "saas",
                "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                "awesome-org.com",
                [],
                "self-managed",
            ),
            (
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Host-Name": "gitlab.com",
                    "X-Gitlab-Saas-Namespace-Ids": "1",
                },
                "saas",
                "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                "gitlab.com",
                ["1"],
                "saas",
            ),
            (
                {
                    "User-Agent": "vs-code",
                },
                None,
                "",
                "",
                "",
                [],
                "",
            ),
        ],
    )
    def test_track_snowplow_event(
        self,
        request_headers,
        jwt_realm_claim,
        expected_instance_id,
        expected_user_id,
        expected_gitlab_host_name,
        expected_gitlab_saas_namespace_ids,
        expected_realm,
    ):
        mock_request = mock.Mock(spec=Request)

        mock_instrumentator = mock.Mock(spec=SnowplowInstrumentator)
        mock_request.headers = request_headers
        mock_request.user.claims.gitlab_realm = jwt_realm_claim

        telemetry_1 = Telemetry(
            requests=1,
            accepts=2,
            errors=3,
            lang="python",
            model_engine="vertex",
            model_name="code-gecko",
        )
        telemetry_2 = Telemetry(
            requests=4,
            accepts=5,
            errors=6,
            lang="golang",
            model_engine="vertex",
            model_name="text-bison",
        )

        test_telemetry = [telemetry_1, telemetry_2]

        suggestion_request = SuggestionsRequest(
            current_file=CurrentFile(
                content_above_cursor="123",
                content_below_cursor="123456",
                file_name="foobar.py",
            ),
            telemetry=test_telemetry,
        )
        track_snowplow_event(
            req=mock_request,
            payload=suggestion_request,
            snowplow_instrumentator=mock_instrumentator,
        )

        mock_instrumentator.watch.assert_called_once()
        args = mock_instrumentator.watch.call_args[1]
        assert len(args) == 10
        assert len(args["telemetry"]) == 2
        assert args["telemetry"][0].__dict__ == telemetry_1.__dict__
        assert args["telemetry"][1].__dict__ == telemetry_2.__dict__
        assert args["prefix_length"] == 3
        assert args["suffix_length"] == 6
        assert args["language"] == "python"
        assert args["user_agent"] == "vs-code"
        assert args["gitlab_realm"] == expected_realm
        assert args["gitlab_instance_id"] == expected_instance_id
        assert args["gitlab_global_user_id"] == expected_user_id
        assert args["gitlab_host_name"] == expected_gitlab_host_name
        assert args["gitlab_saas_namespace_ids"] == expected_gitlab_saas_namespace_ids


class TestCodeGenerations:
    @pytest.mark.parametrize(
        (
            "prompt_version",
            "prefix",
            "prompt",
            "model_provider",
            "model_name",
            "model_output_text",
            "want_vertex_called",
            "want_anthropic_called",
            "want_vertex_prompt_prepared_called",
            "want_anthropic_prompt_prepared_called",
            "want_status",
            "want_prompt",
            "want_choices",
        ),
        [
            (
                1,
                "foo",
                None,
                "vertex-ai",
                "code-bison@002",
                "foo",
                True,
                False,
                False,
                False,
                200,
                None,
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),  # v1 without prompt - vertex-ai
            (
                1,
                "foo",
                None,
                "anthropic",
                "claude-2.1",
                "foo",
                False,
                True,
                False,
                False,
                200,
                None,
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),  # v1 without prompt - anthropic
            (
                1,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                "foo",
                True,
                False,
                False,
                False,
                200,
                None,
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),  # v1 with prompt - vertex-ai
            (
                1,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                "foo",
                True,
                False,
                False,
                False,
                200,
                None,
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),  # v1 with prompt - anthropic
            (
                1,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                "",
                True,
                False,
                False,
                False,
                200,
                None,
                [],
            ),  # v1 empty suggestions from model
            (
                2,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                "foo",
                True,
                False,
                True,
                False,
                200,
                "bar",
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),  # v2 with prompt - vertex-ai
            (
                2,
                "foo",
                "bar",
                "anthropic",
                "claude-2.0",
                "foo",
                False,
                True,
                False,
                True,
                200,
                "bar",
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),  # v2 with prompt - anthropic
            (
                2,
                "foo",
                None,
                "anthropic",
                "claude-2.0",
                "foo",
                False,
                False,
                False,
                False,
                422,
                None,
                None,
            ),  # v2 without prompt field
            (
                2,
                "foo",
                "bar",
                "anthropic",
                "claude-2.1",
                "",
                False,
                True,
                False,
                True,
                200,
                "bar",
                [],
            ),  # v2 empty suggestions from model
        ],
    )
    def test_non_stream_response(
        self,
        mock_client,
        prompt_version,
        prefix,
        prompt,
        model_provider,
        model_name,
        model_output_text,
        want_vertex_called,
        want_anthropic_called,
        want_vertex_prompt_prepared_called,
        want_anthropic_prompt_prepared_called,
        want_status,
        want_prompt,
        want_choices,
    ):
        model_output = CodeSuggestionsOutput(
            text=model_output_text,
            score=0,
            model=ModelMetadata(name="some-model", engine="some-engine"),
            lang_id=LanguageId.PYTHON,
        )

        code_generations_vertex_mock = mock.Mock(spec=CodeGenerations)
        code_generations_vertex_mock.execute = mock.AsyncMock(return_value=model_output)

        code_generations_anthropic_mock = mock.Mock(spec=CodeGenerations)
        code_generations_anthropic_mock.execute = mock.AsyncMock(
            return_value=model_output
        )
        container = CodeSuggestionsContainer()

        with container.code_generations_vertex.override(
            code_generations_vertex_mock
        ), container.code_generations_anthropic.override(
            code_generations_anthropic_mock
        ):
            response = mock_client.post(
                "/v2/code/generations",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json={
                    "prompt_version": prompt_version,
                    "project_path": "gitlab-org/gitlab",
                    "project_id": 278964,
                    "current_file": {
                        "file_name": "main.py",
                        "content_above_cursor": prefix,
                        "content_below_cursor": "\n",
                    },
                    "prompt": prompt,
                    "model_provider": model_provider,
                    "model_name": model_name,
                },
            )

        assert response.status_code == want_status
        assert code_generations_vertex_mock.execute.called == want_vertex_called
        assert code_generations_anthropic_mock.execute.called == want_anthropic_called

        if want_vertex_prompt_prepared_called:
            code_generations_vertex_mock.with_prompt_prepared.assert_called_with(
                want_prompt
            )

        if want_anthropic_prompt_prepared_called:
            code_generations_anthropic_mock.with_prompt_prepared.assert_called_with(
                want_prompt
            )

        if want_status == 200:
            body = response.json()
            assert body["choices"] == want_choices

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
        container = CodeSuggestionsContainer()

        with container.code_generations_anthropic.override(code_generations_mock):
            response = mock_client.post(
                "/v2/code/generations",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json={
                    "prompt_version": 2,
                    "project_path": "gitlab-org/gitlab",
                    "project_id": 278964,
                    "current_file": {
                        "file_name": "main.py",
                        "content_above_cursor": "# create function",
                        "content_below_cursor": "\n",
                    },
                    "prompt": "# create a function",
                    "model_provider": "anthropic",
                },
            )

        assert response.status_code == 200
        assert response.text == expected_response
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    @pytest.mark.parametrize("path", ["/v2/completions", "/v2/code/generations"])
    def test_failed_authorization_scope(self, mock_client, path):
        response = mock_client.post(
            path,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "prompt_version": 1,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "# Create a fast binary search\n",
                    "content_below_cursor": "\n",
                },
            },
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Forbidden"}
