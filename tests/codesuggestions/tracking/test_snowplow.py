import json
from unittest import mock

import pytest
from snowplow_tracker import Snowplow, StructuredEvent

from codesuggestions.tracking import (
    SnowplowClient,
    SnowplowClientConfiguration,
    SnowplowEvent,
    SnowplowEventContext,
)


class TestSnowplowClient:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_initialization(self, mock_emitter_init, mock_tracker_init):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        configuration = SnowplowClientConfiguration(
            namespace="gl",
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
        )
        SnowplowClient(configuration)

        mock_emitter_init.assert_called_once()
        mock_tracker_init.assert_called_once()

        emitter_args = mock_emitter_init.call_args[1]
        assert emitter_args["batch_size"] == 1
        assert emitter_args["thread_count"] == 5
        assert emitter_args["endpoint"] == configuration.endpoint

        tracker_args = mock_tracker_init.call_args[1]
        assert tracker_args["app_id"] == configuration.app_id
        assert tracker_args["namespace"] == configuration.namespace
        assert len(tracker_args["emitters"]) == 1

    @mock.patch("snowplow_tracker.events.StructuredEvent.__init__")
    @mock.patch("snowplow_tracker.Tracker.track")
    def test_track(self, mock_track, mock_structured_event_init):
        mock_structured_event_init.return_value = None

        configuration = SnowplowClientConfiguration(
            namespace="gl",
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
        )
        context = SnowplowEventContext(
            suggestions_shown=1,
            suggestions_failed=0,
            suggestions_accepted=1,
            prefix_length=2048,
            suffix_length=1024,
            language="python",
            user_agent="vs-code-gitlab-workflow",
            gitlab_realm="saas",
        )
        event = SnowplowEvent(
            context=context,
            category="code_suggestions",
            action="suggestion_requested",
        )
        SnowplowClient(configuration).track(event)

        mock_structured_event_init.assert_called_once()

        init_args = mock_structured_event_init.call_args[1]
        assert init_args["category"] == event.category
        assert init_args["action"] == event.action

        context_data = init_args["context"][0].to_json()["data"]
        assert context_data == event.context._asdict()

        mock_track.assert_called_once()
