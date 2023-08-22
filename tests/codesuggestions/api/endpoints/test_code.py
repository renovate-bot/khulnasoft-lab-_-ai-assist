from dataclasses import asdict
from unittest import mock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from snowplow_tracker import Snowplow

from codesuggestions import Config
from codesuggestions.api import create_fast_api_server
from codesuggestions.api.v2.api import api_router
from codesuggestions.api.v2.endpoints.code import (
    CurrentFile,
    SuggestionsRequest,
    track_snowplow_event,
)
from codesuggestions.deps import CodeSuggestionsContainer
from codesuggestions.experimentation.base import ExperimentTelemetry
from codesuggestions.instrumentators.base import Telemetry
from codesuggestions.suggestions.processing.base import ModelEngineOutput
from codesuggestions.suggestions.processing.typing import (
    LanguageId,
    MetadataCodeContent,
    MetadataModel,
    MetadataPromptBuilder,
)
from codesuggestions.tracking.instrumentator import SnowplowInstrumentator


class TestCodeCompletions:
    @pytest.fixture(scope="class")
    def client(self):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        yield client

    def test_successful_response(self, client):
        model_output = ModelEngineOutput(
            text="def search",
            model=MetadataModel(name="code-gecko", engine="vertex-ai"),
            lang_id=LanguageId.PYTHON,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                },
                experiments=[ExperimentTelemetry(name="truncate_suffix", variant=1)],
            ),
        )

        code_completions_mock = mock.AsyncMock(return_value=model_output)
        container = CodeSuggestionsContainer()

        with container.code_completions.override(code_completions_mock):
            response = client.post(
                "/v2/completions",
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
        assert body["choices"][0]["text"] == "def search"
        assert body["experiments"] == [{"name": "truncate_suffix", "variant": 1}]
        assert body["model"] == {
            "engine": "vertex-ai",
            "lang": "python",
            "name": "code-gecko",
        }


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
