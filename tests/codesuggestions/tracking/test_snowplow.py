from unittest import mock

import json
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

    @mock.patch("snowplow_tracker.Snowplow.create_tracker")
    def test_initialization(self, mock_create):
        configuration = SnowplowClientConfiguration(
            namespace="gl",
            endpoint="whitechoc.local",
            app_id="gitlab_ai_gateway",
        )
        SnowplowClient(configuration)

        mock_create.assert_called_once()

        create_args = mock_create.call_args[1]
        assert create_args["app_id"] == configuration.app_id
        assert create_args["namespace"] == configuration.namespace
        assert create_args["endpoint"] == configuration.endpoint
        assert create_args["emitter_config"].batch_size == 1

    @mock.patch("snowplow_tracker.events.StructuredEvent.__init__")
    @mock.patch("snowplow_tracker.Tracker.track")
    def test_track(self, mock_track, mock_structured_event_init):
        mock_structured_event_init.return_value = None

        configuration = SnowplowClientConfiguration(
            namespace="gl",
            endpoint="wonderful.chocolate.factory",
            app_id="gitlab_ai_gateway",
        )
        context = SnowplowEventContext(
            suggestions_shown=1,
            suggestions_failed=0,
            suggestions_accepted=1,
            prefix_length=2048,
            suffix_length=1024,
            language="Python",
            user_agent="VSCode",
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
