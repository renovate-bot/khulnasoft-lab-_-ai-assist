from dataclasses import asdict
from unittest import mock

import pytest
from snowplow_tracker import SelfDescribingJson, Snowplow, StructuredEvent

from ai_gateway.tracking import (
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
            batch_size=3,
            thread_count=2,
        )
        SnowplowClient(configuration)

        mock_emitter_init.assert_called_once()
        mock_tracker_init.assert_called_once()

        emitter_args = mock_emitter_init.call_args[1]
        assert emitter_args["batch_size"] == 3
        assert emitter_args["thread_count"] == 2
        assert emitter_args["endpoint"] == configuration.endpoint

        tracker_args = mock_tracker_init.call_args[1]
        assert tracker_args["app_id"] == configuration.app_id
        assert tracker_args["namespace"] == configuration.namespace
        assert len(tracker_args["emitters"]) == 1

    @pytest.mark.parametrize(
        ("inputs"),
        [
            (
                SnowplowEvent(
                    context=None,
                    category="code_suggestions",
                    action="suggestion_requested",
                    label="some label",
                    value=1,
                )
            ),
            (
                SnowplowEvent(
                    category="code_suggestions",
                    action="suggestion_requested",
                    label="some label",
                    value=1,
                    context=SnowplowEventContext(
                        prefix_length=2048,
                        suffix_length=1024,
                        language="python",
                        gitlab_realm="saas",
                        is_direct_connection=True,
                        gitlab_instance_id="ABCDEF",
                        gitlab_global_user_id="123XYZ",
                        gitlab_host_name="gitlab.com",
                        gitlab_saas_duo_pro_namespace_ids=[54321],
                    ),
                )
            ),
        ],
    )
    def test_track(self, inputs):
        with mock.patch("snowplow_tracker.Tracker.track") as mock_track, mock.patch(
            "snowplow_tracker.events.StructuredEvent.__init__"
        ) as mock_structured_event_init:
            configuration = SnowplowClientConfiguration(
                namespace="gl",
                endpoint="https://whitechoc.local",
                app_id="gitlab_ai_gateway",
            )
            mock_structured_event_init.return_value = None
            SnowplowClient(configuration).track(event=inputs)

            mock_track.assert_called_once()
            mock_structured_event_init.assert_called_once()
            event_init_args = mock_structured_event_init.call_args[1]

            assert event_init_args["value"] == inputs.value
            assert event_init_args["category"] == inputs.category
            assert event_init_args["action"] == inputs.action
            assert event_init_args["label"] == inputs.label

            if inputs.context is None:
                return

            context = event_init_args["context"][0]
            assert isinstance(context, SelfDescribingJson)
            assert context.to_json()["schema"] == SnowplowClient.SCHEMA
            assert context.to_json()["data"] == asdict(inputs.context)


class TestSnowplowInstrumentator:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @pytest.mark.parametrize(
        (
            "inputs",
            "expectations",
        ),
        [
            (
                SnowplowEvent(
                    context=None,
                    category="code_suggestions",
                    action="suggestion_requested",
                    label="some label",
                    value=1,
                ),
                SnowplowEvent(
                    context=None,
                    category="code_suggestions",
                    action="suggestion_requested",
                    label="some label",
                    value=1,
                ),
            )
        ],
    )
    def test_watch(self, inputs, expectations):
        mock_client = mock.Mock(spec=SnowplowClient)
        instrumentator = SnowplowInstrumentator(client=mock_client)

        instrumentator.watch(inputs)

        mock_client.track.assert_called_once()

        assert mock_client.track.call_args[0][0] == expectations
