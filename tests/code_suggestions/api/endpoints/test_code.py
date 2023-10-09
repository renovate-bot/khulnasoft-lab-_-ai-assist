from unittest import mock

import pytest
from fastapi import Request
from snowplow_tracker import Snowplow

from ai_gateway.api.v2.api import api_router
from ai_gateway.api.v2.endpoints.code import (
    CurrentFile,
    SuggestionsRequest,
    track_snowplow_event,
)
from ai_gateway.auth import User, UserClaims
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
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
from tests.fixtures.fast_api import *


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


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
                    model=ModelMetadata(name="claude-instant-1", engine="anthropic"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-instant-1",
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
                    model=ModelMetadata(name="claude-instant-1", engine="anthropic"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-instant-1",
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
                    model=ModelMetadata(name="claude-instant-1", engine="anthropic"),
                    lang_id=LanguageId.PYTHON,
                    metadata=CodeSuggestionsOutput.Metadata(
                        experiments=[],
                    ),
                ),
                {
                    "id": "id",
                    "model": {
                        "engine": "anthropic",
                        "name": "claude-instant-1",
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
            current_file["content_above_cursor"],
            current_file["content_below_cursor"],
            current_file["file_name"],
            current_file.get("language_identifier", None),
            **code_completions_kwargs,
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
            "expected_realm",
        ),
        [
            (
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Realm": "saas",
                },
                None,
                "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                "saas",
            ),
            (
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Realm": "self-managed",
                },
                "saas",
                "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                "self-managed",
            ),
            (
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                },
                "saas",
                "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
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
            ),
        ],
    )
    def test_track_snowplow_event(
        self,
        request_headers,
        jwt_realm_claim,
        expected_instance_id,
        expected_user_id,
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
        assert len(args) == 8
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


class TestCodeGenerations:
    @pytest.mark.parametrize(
        (
            "prompt_version",
            "prefix",
            "prompt",
            "model_output_text",
            "want_called",
            "want_status",
            "want_prompt",
            "want_choices",
        ),
        [
            (
                1,
                "foo",
                None,
                "foo",
                True,
                200,
                None,
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),
            (
                1,
                "foo",
                "bar",
                "foo",
                True,
                200,
                None,
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),
            (
                1,
                "foo",
                "bar",
                "",
                True,
                200,
                None,
                [],
            ),  # v1 empty suggestions from model
            (
                2,
                "foo",
                "bar",
                "foo",
                True,
                200,
                "bar",
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),
            (
                2,
                "foo",
                None,
                "foo",
                False,
                422,
                None,
                None,
            ),  # v2 request need the prompt field
            (
                2,
                "foo",
                "bar",
                "",
                True,
                200,
                "bar",
                [],
            ),  # v2 empty suggestions from model
        ],
    )
    def test_request_versioning(
        self,
        mock_client,
        prompt_version,
        prefix,
        prompt,
        model_output_text,
        want_called,
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

        code_generations_mock = mock.Mock(spec=CodeGenerations)
        code_generations_mock.execute = mock.AsyncMock(return_value=model_output)
        container = CodeSuggestionsContainer()

        with container.code_generations_vertex.override(code_generations_mock):
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
                },
            )

        assert response.status_code == want_status
        assert code_generations_mock.execute.called == want_called

        if code_generations_mock.with_prompt_prepared.called:
            code_generations_mock.with_prompt_prepared.assert_called_with(want_prompt)

        if want_status == 200:
            body = response.json()
            assert body["choices"] == want_choices


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(
                is_third_party_ai_default=False, scopes=["unauthorized_scope"]
            ),
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
