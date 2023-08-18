from dataclasses import asdict
from unittest import mock

import pytest
from snowplow_tracker import Snowplow

from codesuggestions.instrumentators.base import Telemetry
from codesuggestions.tracking import (
    RequestCount,
    SnowplowClient,
    SnowplowClientConfiguration,
    SnowplowEvent,
    SnowplowEventContext,
    SnowplowInstrumentator,
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
            request_counts=[
                RequestCount(
                    requests=1,
                    errors=0,
                    accepts=1,
                    lang="python",
                    model_engine="vertex-ai",
                    model_name="code-gecko",
                )
            ],
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
        assert context_data == asdict(event.context)

        mock_track.assert_called_once()


class TestSnowplowInstrumentator:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    def test_watch(self):
        mock_client = mock.Mock(spec=SnowplowClient)
        instrumentator = SnowplowInstrumentator(client=mock_client)

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

        instrumentator.watch(
            telemetry=test_telemetry,
            prefix_length=11,
            suffix_length=22,
            language="ruby",
            user_agent="vs-code",
            gitlab_realm="saas",
        )

        mock_client.track.assert_called_once()

        event = mock_client.track.call_args[0][0].context
        assert len(event.request_counts) == 2
        assert event.request_counts[0].__dict__ == telemetry_1.__dict__
        assert event.request_counts[1].__dict__ == telemetry_2.__dict__
        assert event.prefix_length == 11
        assert event.suffix_length == 22
        assert event.language == "ruby"
        assert event.user_agent == "vs-code"
        assert event.gitlab_realm == "saas"
