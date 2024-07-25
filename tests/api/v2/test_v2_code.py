import time
from typing import Dict, List, Union
from unittest.mock import Mock, patch

import pytest
from dependency_injector import containers
from fastapi import status
from fastapi.testclient import TestClient
from snowplow_tracker import Snowplow
from starlette.datastructures import CommaSeparatedStrings
from structlog.testing import capture_logs

from ai_gateway.api.v2 import api_router
from ai_gateway.auth import User, UserClaims
from ai_gateway.config import Config
from ai_gateway.models.base_chat import Message, Role
from ai_gateway.tracking.container import ContainerTracking
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import SnowplowEvent, SnowplowEventContext


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def auth_user():
    return User(
        authenticated=True,
        claims=UserClaims(
            scopes=["code_suggestions"],
            subject="1234",
            gitlab_realm="self-managed",
            issuer="issuer",
        ),
    )


@pytest.fixture
def mock_config():
    config = Config()
    config.custom_models.enabled = True

    yield config


class TestCodeCompletions:
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @pytest.mark.parametrize(
        ("mock_completions_legacy_output_texts", "expected_response"),
        [
            # non-empty suggestions from model
            (
                ["def search", "println"],
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
                        },
                        {
                            "text": "println",
                            "index": 0,
                            "finish_reason": "length",
                        },
                    ],
                },
            ),
            # empty suggestions from model
            (
                [""],
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
        mock_completions_legacy: Mock,
        expected_response: dict,
    ):
        response = mock_client.post(
            "/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
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
        ("prompt_version", "mock_suggestions_output_text", "expected_response"),
        [
            # non-empty suggestions from model
            (
                1,
                "def search",
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
                "def search",
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
                "",
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
        mock_completions: Mock,
        expected_response: dict,
    ):
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

        response = mock_client.post(
            "/completions",
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

        assert body["choices"] == expected_response["choices"]
        assert body["model"] == expected_response["model"]

        mock_completions.assert_called_with(
            prefix=current_file["content_above_cursor"],
            suffix=current_file["content_below_cursor"],
            file_name=current_file["file_name"],
            editor_lang=current_file.get("language_identifier", None),
            stream=False,
            **code_completions_kwargs,
        )

    @pytest.mark.parametrize("prompt_version", [1])
    def test_request_latency(
        self,
        prompt_version: int,
        mock_client: TestClient,
        mock_completions: Mock,
    ):
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

        # get valid duration
        with capture_logs() as cap_logs:
            response = mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Rails-Send-Start": str(time.time() - 1),
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                },
                json=data,
            )
            assert response.status_code == 200
            assert len(cap_logs) == 3
            assert cap_logs[-1]["status_code"] == 200
            assert cap_logs[-1]["method"] == "POST"
            assert cap_logs[-1]["duration_request"] >= 1

        # -1 in duration_request indicates invalid X-Gitlab-Rails-Send-Start header
        with capture_logs() as cap_logs:
            response = mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-Gitlab-Rails-Send-Start": "invalid epoch time",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                },
                json=data,
            )
            assert response.status_code == 200
            assert len(cap_logs) == 3
            assert cap_logs[-1]["status_code"] == 200
            assert cap_logs[-1]["method"] == "POST"
            assert cap_logs[-1]["duration_request"] == -1

        # -1 in duration_request indicates missing X-Gitlab-Rails-Send-Start header
        with capture_logs() as cap_logs:
            response = mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                },
                json=data,
            )
            assert response.status_code == 200
            assert len(cap_logs) == 3
            assert cap_logs[-1]["status_code"] == 200
            assert cap_logs[-1]["method"] == "POST"
            assert cap_logs[-1]["duration_request"] == -1

    @pytest.mark.parametrize(
        (
            "prompt_version",
            "prompt",
            "model_provider",
            "model_name",
            "model_endpoint",
            "model_api_key",
            "want_litellm_called",
            "want_agent_called",
            "want_status",
            "want_choices",
        ),
        [
            # test litellm completions
            (
                2,
                "foo",
                "litellm",
                "codegemma",
                "http://localhost:4000/",
                "api-key",
                True,
                False,
                200,
                [{"text": "test completion", "index": 0, "finish_reason": "length"}],
            ),
            (
                2,
                "",
                "litellm",
                "codegemma",
                "http://localhost:4000/",
                "api-key",
                False,
                True,
                200,
                [{"text": "test completion", "index": 0, "finish_reason": "length"}],
            ),
            (
                2,
                None,
                "litellm",
                "codegemma",
                "http://localhost:4000/",
                "api-key",
                False,
                True,
                200,
                [{"text": "test completion", "index": 0, "finish_reason": "length"}],
            ),
        ],
    )
    def test_non_stream_response(
        self,
        mock_client,
        mock_llm_text: Mock,
        mock_agent_model: Mock,
        prompt_version,
        prompt,
        model_provider,
        model_name,
        model_endpoint,
        model_api_key,
        want_litellm_called,
        want_agent_called,
        want_status,
        want_choices,
    ):
        response = mock_client.post(
            "/code/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_version": prompt_version,
                "project_path": "gitlab-org/gitlab",
                "project_id": 278964,
                "current_file": {
                    "file_name": "main.py",
                    "content_above_cursor": "foo",
                    "content_below_cursor": "\n",
                },
                "prompt": prompt,
                "model_provider": model_provider,
                "model_name": model_name,
                "model_endpoint": model_endpoint,
                "model_api_key": model_api_key,
            },
        )

        assert response.status_code == want_status
        assert mock_llm_text.called == want_litellm_called
        assert mock_agent_model.called == want_agent_called

        if want_status == 200:
            body = response.json()
            assert body["choices"] == want_choices

    @pytest.mark.parametrize(
        ("expected_response", "provider", "model"),
        [
            (
                "def search",
                "anthropic",
                None,
            ),
            (
                "def search",
                "litellm",
                "mistral",
            ),
        ],
    )
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_completions_stream: Mock,
        expected_response: str,
        provider: str,
        model: str | None,
    ):
        current_file = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
        }
        data = {
            "prompt_version": 1,
            "project_path": "gitlab-org/gitlab",
            "project_id": 278964,
            "model_provider": provider,
            "current_file": current_file,
            "stream": True,
        }
        if model:
            data["model_name"] = model

        response = mock_client.post(
            "/completions",
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

        mock_completions_stream.assert_called_with(
            prefix=current_file["content_above_cursor"],
            suffix=current_file["content_below_cursor"],
            file_name=current_file["file_name"],
            editor_lang=current_file.get("language_identifier", None),
            stream=True,
        )

    @pytest.mark.parametrize(
        (
            "auth_user",
            "telemetry",
            "current_file",
            "request_headers",
            "expected_language",
        ),
        [
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
                    "X-Gitlab-Instance-Id": "1234",
                    "X-Gitlab-Global-User-Id": "1234",
                    "X-Gitlab-Host-Name": "gitlab.com",
                    "X-Gitlab-Saas-Namespace-Ids": "1,2,3",
                    "X-Gitlab-Saas-Duo-Pro-Namespace-Ids": "4,5,6",
                    "X-Gitlab-Realm": "self-managed",
                },
                "python",
            )
        ],
    )
    def test_snowplow_tracking(
        self,
        mock_client: TestClient,
        mock_container: containers.Container,
        mock_completions_legacy: Mock,
        auth_user: User,
        telemetry: List[Dict[str, Union[str, int, None]]],
        current_file: Dict[str, str],
        expected_language: str,
        request_headers: Dict[str, str],
    ):
        expected_event = SnowplowEvent(
            context=SnowplowEventContext(
                prefix_length=len(current_file.get("content_above_cursor", "")),
                suffix_length=len(current_file.get("content_below_cursor", "")),
                language=expected_language,
                user_agent=request_headers.get("User-Agent", ""),
                gitlab_realm=request_headers.get("X-Gitlab-Realm", ""),
                is_direct_connection=True,
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

        snowplow_instrumentator_mock = Mock(spec=SnowplowInstrumentator)

        snowplow_container_mock = Mock(spec=ContainerTracking)
        snowplow_container_mock.instrumentator = Mock(
            return_value=snowplow_instrumentator_mock
        )

        with patch.object(mock_container, "snowplow", snowplow_container_mock):
            mock_client.post(
                "/completions",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                    **request_headers,
                },
                json={
                    "prompt_version": 1,
                    "project_path": "gitlab-org/gitlab",
                    "project_id": 278964,
                    "current_file": current_file,
                    "telemetry": telemetry,
                    "choices_count": 1,
                },
            )

        snowplow_instrumentator_mock.watch.assert_called_once()
        args = snowplow_instrumentator_mock.watch.call_args[0]
        assert len(args) == 1
        assert args[0] == expected_event

    @pytest.mark.parametrize(
        ("auth_user", "extra_headers", "expected_status_code"),
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
                {"X-GitLab-Instance-Id": "1234"},
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
                {"X-Gitlab-Global-User-Id": "1234"},
                200,
            ),
        ],
    )
    def test_successful_response_with_correct_issuers(
        self,
        mock_client: TestClient,
        mock_completions: Mock,
        auth_user: User,
        extra_headers: Dict[str, str],
        expected_status_code: int,
    ):
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
        }

        response = mock_client.post(
            "/completions",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Realm": "self-managed",
            }
            | extra_headers,
            json=data,
        )

        assert response.status_code == expected_status_code


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
            "model_endpoint",
            "model_api_key",
            "mock_output_text",
            "want_vertex_called",
            "want_anthropic_called",
            "want_anthropic_chat_called",
            "want_prompt_prepared_called",
            "want_litellm_called",
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
                None,
                None,
                "foo",
                True,
                False,
                False,
                False,
                False,
                200,
                None,
                [{"text": "\nfoo", "index": 0, "finish_reason": "length"}],
            ),  # v1 without prompt - vertex-ai
            (
                1,
                "foo",
                None,
                "anthropic",
                "claude-2.1",
                None,
                None,
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
                None,
                None,
                "foo",
                True,
                False,
                False,
                False,
                False,
                200,
                None,
                [{"text": "\nfoo", "index": 0, "finish_reason": "length"}],
            ),  # v1 with prompt - vertex-ai
            (
                1,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                None,
                None,
                "foo",
                True,
                False,
                False,
                False,
                False,
                200,
                None,
                [{"text": "\nfoo", "index": 0, "finish_reason": "length"}],
            ),  # v1 with prompt - anthropic
            (
                1,
                "foo",
                "bar",
                "vertex-ai",
                "code-bison@002",
                None,
                None,
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
                None,
                None,
                "foo",
                True,
                False,
                False,
                True,
                False,
                200,
                "bar",
                [{"text": "\nfoo", "index": 0, "finish_reason": "length"}],
            ),  # v2 with prompt - vertex-ai
            (
                2,
                "foo",
                "bar",
                "anthropic",
                "claude-2.0",
                None,
                None,
                "foo",
                False,
                True,
                False,
                True,
                False,
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
                None,
                None,
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
                None,
                None,
                "",
                False,
                True,
                False,
                True,
                False,
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
                None,
                None,
                "foo",
                False,
                False,
                True,
                True,
                False,
                200,
                [
                    Message(role=Role.SYSTEM, content="foo"),
                    Message(role=Role.USER, content="bar"),
                ],
                [{"text": "foo", "index": 0, "finish_reason": "length"}],
            ),  # v3 with prompt - anthropic
            (
                3,
                "foo",
                [
                    {"role": "system", "content": "foo"},
                    {"role": "user", "content": "bar"},
                ],
                "litellm",
                "mistral",
                "http://localhost:11434/v1",
                "api-key",
                "foo",
                False,
                False,
                False,
                False,
                True,
                200,
                [
                    Message(role=Role.SYSTEM, content="foo"),
                    Message(role=Role.USER, content="bar"),
                ],
                [{"text": "\nfoo", "index": 0, "finish_reason": "length"}],
            ),  # v3 with prompt - litellm
        ],
    )
    def test_non_stream_response(
        self,
        mock_client,
        mock_container: containers.Container,
        mock_code_bison: Mock,
        mock_anthropic: Mock,
        mock_anthropic_chat: Mock,
        mock_llm_chat: Mock,
        mock_with_prompt_prepared: Mock,
        prompt_version,
        prefix,
        prompt,
        model_provider,
        model_name,
        model_endpoint,
        model_api_key,
        mock_output_text,
        want_vertex_called,
        want_anthropic_called,
        want_anthropic_chat_called,
        want_prompt_prepared_called,
        want_litellm_called,
        want_status,
        want_prompt,
        want_choices,
    ):
        response = mock_client.post(
            "/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
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
                "model_endpoint": model_endpoint,
                "model_api_key": model_api_key,
            },
        )

        assert response.status_code == want_status
        assert mock_code_bison.called == want_vertex_called
        assert mock_anthropic.called == want_anthropic_called
        assert mock_llm_chat.called == want_litellm_called
        assert mock_anthropic_chat.called == want_anthropic_chat_called

        if want_prompt_prepared_called:
            mock_with_prompt_prepared.assert_called_with(want_prompt)

        if want_status == 200:
            body = response.json()
            assert body["choices"] == want_choices

    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_generations_stream: Mock,
        mock_suggestions_output_text: str,
    ):
        response = mock_client.post(
            "/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
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
        assert response.text == mock_suggestions_output_text
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
                    "X-Gitlab-Instance-Id": "1234",
                    "X-Gitlab-Global-User-Id": "XTuMnZ6XTWkP3yh0ZwXualmOZvm2Gg/bk9jyfkL7Y6k=",
                    "X-Gitlab-Host-Name": "gitlab.com",
                    "X-Gitlab-Saas-Namespace-Ids": "1,2,3",
                    "X-Gitlab-Saas-Duo-Pro-Namespace-Ids": "4,5,6",
                    "X-Gitlab-Realm": "self-managed",
                },
                "python",
            )
        ],
    )
    def test_snowplow_tracking(
        self,
        mock_client: TestClient,
        mock_container: containers.Container,
        mock_generations: Mock,
        telemetry: List[Dict[str, Union[str, int, None]]],
        current_file: Dict[str, str],
        expected_language: str,
        request_headers: Dict[str, str],
    ):
        expected_event = SnowplowEvent(
            context=SnowplowEventContext(
                prefix_length=len(current_file.get("content_above_cursor", "")),
                suffix_length=len(current_file.get("content_below_cursor", "")),
                language=expected_language,
                user_agent=request_headers.get("User-Agent", ""),
                gitlab_realm=request_headers.get("X-Gitlab-Realm", ""),
                is_direct_connection=False,
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

        snowplow_instrumentator_mock = Mock(spec=SnowplowInstrumentator)

        snowplow_container_mock = Mock(spec=ContainerTracking)
        snowplow_container_mock.instrumentator = Mock(
            return_value=snowplow_instrumentator_mock
        )

        with patch.object(mock_container, "snowplow", snowplow_container_mock):
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
                    "choices_count": 1,
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
            claims=UserClaims(
                scopes=["unauthorized_scope"],
                subject="1234",
                gitlab_realm="self-managed",
            ),
        )

    @pytest.mark.parametrize(
        ("path", "error_message"),
        [
            ("/completions", "Unauthorized to access code completions"),
            ("/code/generations", "Unauthorized to access code generations"),
        ],
    )
    def test_failed_authorization_scope(self, mock_client, path, error_message):
        response = mock_client.post(
            path,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
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

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json() == {"detail": error_message}


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
            "/code/generations",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-GitLab-Realm": "self-managed",
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

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json() == {"detail": "Unauthorized to access code generations"}
