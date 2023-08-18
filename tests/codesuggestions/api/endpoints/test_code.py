from dataclasses import asdict
from unittest import mock

import pytest
from fastapi import Request
from snowplow_tracker import Snowplow

from codesuggestions.api.v2.endpoints.code import (
    CurrentFile,
    SuggestionsRequest,
    track_snowplow_event,
)
from codesuggestions.instrumentators.base import Telemetry
from codesuggestions.tracking.instrumentator import SnowplowInstrumentator


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
