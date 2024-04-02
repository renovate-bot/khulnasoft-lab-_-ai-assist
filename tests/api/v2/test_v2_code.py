from typing import AsyncIterator, Dict, List, Union
from unittest import mock

import pytest
from dependency_injector import containers
from fastapi.testclient import TestClient
from snowplow_tracker import Snowplow
from starlette.datastructures import CommaSeparatedStrings

from ai_gateway.api.v2 import api_router
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
from ai_gateway.experimentation.base import ExperimentTelemetry
from ai_gateway.models import ModelMetadata
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.models.chat_model_base import Message, Role
from ai_gateway.tracking.container import ContainerTracking
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import (
    RequestCount,
    SnowplowEvent,
    SnowplowEventContext,
)


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
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

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
                    tokens_consumption_metadata=TokensConsumptionMetadata(
                        input_tokens=0, output_tokens=0
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
                    tokens_consumption_metadata=TokensConsumptionMetadata(
                        input_tokens=0, output_tokens=0
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
        mock_container: containers.DeclarativeContainer,
        model_output: ModelEngineOutput,
        expected_response: dict,
    ):
        code_completions_mock = mock.Mock(spec=CodeCompletionsLegacy)
        code_completions_mock.execute = mock.AsyncMock(return_value=model_output)
        with mock_container.code_suggestions.completions.vertex_legacy.override(
            code_completions_mock
        ):
            response = mock_client.post(
                "/completions",
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
        mock_container: containers.DeclarativeContainer,
        model_output: CodeSuggestionsOutput,
        expected_response: dict,
    ):
        code_completions_mock = mock.Mock(spec=CodeCompletions)
        code_completions_mock.execute = mock.AsyncMock(return_value=model_output)

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

        with mock_container.code_suggestions.completions.anthropic.override(
            code_completions_mock
        ):
            response = mock_client.post(
                "/completions",
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
        mock_container: containers.DeclarativeContainer,
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

        with mock_container.code_suggestions.completions.anthropic.override(
            code_completions_mock
        ):
            response = mock_client.post(
                "/completions",
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

    @pytest.mark.parametrize(
        (
            "telemetry",
            "current_file",
            "request_headers",
            "expected_language",
        ),
        [
            (
                [
                    {
                        "model_engine": "vertex",
                        "model_name": "code-gecko",
                        "requests": 1,
                        "accepts": 1,
                        "errors": 0,
                        "lang": None,
                    }
                ],
                {
                    "file_name": "main.py",
                    "content_above_cursor": "# Create a fast binary search\n",
                    "content_below_cursor": "\n",
                },
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Host-Name": "gitlab.com",
                    "X-Gitlab-Saas-Namespace-Ids": "1,2,3",
                    "X-Gitlab-Saas-Duo-Pro-Namespace-Ids": "4,5,6",
                    "X-Gitlab-Realm": "saas",
                },
                "python",
            )
        ],
    )
    def test_snowplow_tracking(
        self,
        mock_client: TestClient,
        mock_container: containers.DeclarativeContainer,
        telemetry: List[Dict[str, Union[str, int, None]]],
        current_file: Dict[str, str],
        expected_language: str,
        request_headers: Dict[str, str],
    ):
        expected_event = SnowplowEvent(
            context=SnowplowEventContext(
                request_counts=[RequestCount(**rc) for rc in telemetry],
                prefix_length=len(current_file.get("content_above_cursor", "")),
                suffix_length=len(current_file.get("content_below_cursor", "")),
                language=expected_language,
                user_agent=request_headers.get("User-Agent", ""),
                gitlab_realm=request_headers.get("X-Gitlab-Realm", ""),
                gitlab_instance_id=request_headers.get("X-Gitlab-Instance-Id", ""),
                gitlab_global_user_id=request_headers.get(
                    "X-Gitlab-Global-User-Id", ""
                ),
                gitlab_host_name=request_headers.get("X-Gitlab-Host-Name", ""),
                gitlab_saas_namespace_ids=list(
                    CommaSeparatedStrings(
                        request_headers.get("X-Gitlab-Saas-Namespace-Ids", "")
                    )
                ),
                gitlab_saas_duo_pro_namespace_ids=list(
                    CommaSeparatedStrings(
                        request_headers.get("X-Gitlab-Saas-Duo-Pro-Namespace-Ids", "")
                    )
                ),
            )
        )

        model_output = ModelEngineOutput(
            text="def search",
            score=0,
            model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
            lang_id=LanguageId.PYTHON,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                },
                experiments=[ExperimentTelemetry(name="truncate_suffix", variant=1)],
            ),
            tokens_consumption_metadata=TokensConsumptionMetadata(
                input_tokens=0, output_tokens=0
            ),
        )

        code_completions_mock = mock.Mock(spec=CodeCompletionsLegacy)
        code_completions_mock.execute = mock.AsyncMock(return_value=model_output)

        snowplow_instrumentator_mock = mock.Mock(spec=SnowplowInstrumentator)

        snowplow_container_mock = mock.Mock(spec=ContainerTracking)
        snowplow_container_mock.instrumentator = mock.Mock(
            return_value=snowplow_instrumentator_mock
        )

        with mock_container.code_suggestions.completions.vertex_legacy.override(
            code_completions_mock
        ), mock.patch.object(mock_container, "snowplow", snowplow_container_mock):
            mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    **request_headers,
                },
                json={
                    "prompt_version": 1,
                    "project_path": "gitlab-org/gitlab",
                    "project_id": 278964,
                    "current_file": current_file,
                    "telemetry": telemetry,
                },
            )

        snowplow_instrumentator_mock.watch.assert_called_once()
        args = snowplow_instrumentator_mock.watch.call_args[0]
        assert len(args) == 1
        assert args[0] == expected_event


class TestCodeGenerations:
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

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
            "want_anthropic_chat_called",
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
                False,
                True,
                200,
                "bar",
                [],
            ),  # v2 empty suggestions from model
            (
                3,
                "foo",
                [
                    {"role": "system", "content": "foo"},
                    {"role": "user", "content": "bar"},
                ],
                "anthropic",
                "claude-3-opus-20240229",
                "foo",
                False,
                False,
                True,
                False,
                True,
                200,
                [
                    Message(role=Role.SYSTEM, content="foo"),
                    Message(role=Role.USER, content="bar"),
                ],
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),  # v3 with prompt - anthropic
        ],
    )
    def test_non_stream_response(
        self,
        mock_client,
        mock_container: containers.DeclarativeContainer,
        prompt_version,
        prefix,
        prompt,
        model_provider,
        model_name,
        model_output_text,
        want_vertex_called,
        want_anthropic_called,
        want_anthropic_chat_called,
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

        code_generations_anthropic_chat_mock = mock.Mock(spec=CodeGenerations)
        code_generations_anthropic_chat_mock.execute = mock.AsyncMock(
            return_value=model_output
        )

        with mock_container.code_suggestions.generations.vertex.override(
            code_generations_vertex_mock
        ), mock_container.code_suggestions.generations.anthropic_factory.override(
            code_generations_anthropic_mock
        ), mock_container.code_suggestions.generations.anthropic_chat_factory.override(
            code_generations_anthropic_chat_mock
        ):
            response = mock_client.post(
                "/code/generations",
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
        assert (
            code_generations_anthropic_chat_mock.execute.called
            == want_anthropic_chat_called
        )

        if want_vertex_prompt_prepared_called:
            code_generations_vertex_mock.with_prompt_prepared.assert_called_with(
                want_prompt
            )

        if want_anthropic_prompt_prepared_called and want_anthropic_called:
            code_generations_anthropic_mock.with_prompt_prepared.assert_called_with(
                want_prompt
            )

        if want_anthropic_prompt_prepared_called and want_anthropic_chat_called:
            code_generations_anthropic_chat_mock.with_prompt_prepared.assert_called_with(
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
        mock_container: containers.DeclarativeContainer,
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

        with mock_container.code_suggestions.generations.anthropic_factory.override(
            code_generations_mock
        ):
            response = mock_client.post(
                "/code/generations",
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

    @pytest.mark.parametrize(
        (
            "telemetry",
            "current_file",
            "request_headers",
            "expected_language",
        ),
        [
            (
                [
                    {
                        "model_engine": "vertex",
                        "model_name": "code-gecko",
                        "requests": 1,
                        "accepts": 1,
                        "errors": 0,
                        "lang": None,
                    }
                ],
                {
                    "file_name": "main.py",
                    "content_above_cursor": "# Create a fast binary search\n",
                    "content_below_cursor": "\n",
                },
                {
                    "User-Agent": "vs-code",
                    "X-Gitlab-Instance-Id": "9ebada7a-f5e2-477a-8609-17797fa95cb9",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Host-Name": "gitlab.com",
                    "X-Gitlab-Saas-Namespace-Ids": "1,2,3",
                    "X-Gitlab-Saas-Duo-Pro-Namespace-Ids": "4,5,6",
                    "X-Gitlab-Realm": "saas",
                },
                "python",
            )
        ],
    )
    def test_snowplow_tracking(
        self,
        mock_client: TestClient,
        mock_container: containers.DeclarativeContainer,
        telemetry: List[Dict[str, Union[str, int, None]]],
        current_file: Dict[str, str],
        expected_language: str,
        request_headers: Dict[str, str],
    ):
        expected_event = SnowplowEvent(
            context=SnowplowEventContext(
                request_counts=[RequestCount(**rc) for rc in telemetry],
                prefix_length=len(current_file.get("content_above_cursor", "")),
                suffix_length=len(current_file.get("content_below_cursor", "")),
                language=expected_language,
                user_agent=request_headers.get("User-Agent", ""),
                gitlab_realm=request_headers.get("X-Gitlab-Realm", ""),
                gitlab_instance_id=request_headers.get("X-Gitlab-Instance-Id", ""),
                gitlab_global_user_id=request_headers.get(
                    "X-Gitlab-Global-User-Id", ""
                ),
                gitlab_host_name=request_headers.get("X-Gitlab-Host-Name", ""),
                gitlab_saas_namespace_ids=list(
                    CommaSeparatedStrings(
                        request_headers.get("X-Gitlab-Saas-Namespace-Ids", "")
                    )
                ),
                gitlab_saas_duo_pro_namespace_ids=list(
                    CommaSeparatedStrings(
                        request_headers.get("X-Gitlab-Saas-Duo-Pro-Namespace-Ids", "")
                    )
                ),
            )
        )

        model_output = CodeSuggestionsOutput(
            text="some code",
            score=0,
            model=ModelMetadata(name="some-model", engine="some-engine"),
            lang_id=LanguageId.PYTHON,
        )

        code_generations_vertex_mock = mock.Mock(spec=CodeGenerations)
        code_generations_vertex_mock.execute = mock.AsyncMock(return_value=model_output)

        snowplow_instrumentator_mock = mock.Mock(spec=SnowplowInstrumentator)

        snowplow_container_mock = mock.Mock(spec=ContainerTracking)
        snowplow_container_mock.instrumentator = mock.Mock(
            return_value=snowplow_instrumentator_mock
        )

        with mock_container.code_suggestions.generations.vertex.override(
            code_generations_vertex_mock
        ), mock.patch.object(mock_container, "snowplow", snowplow_container_mock):
            response = mock_client.post(
                "/code/generations",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    **request_headers,
                },
                json={
                    "prompt_version": 1,
                    "project_path": "gitlab-org/gitlab",
                    "project_id": 278964,
                    "current_file": current_file,
                    "prompt": "some prompt",
                    "model_provider": "vertex-ai",
                    "model_name": "code-bison@002",
                    "telemetry": telemetry,
                },
            )

        snowplow_instrumentator_mock.watch.assert_called_once()
        args = snowplow_instrumentator_mock.watch.call_args[0]
        assert len(args) == 1
        assert args[0] == expected_event


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return User(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    @pytest.mark.parametrize("path", ["/completions", "/code/generations"])
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
