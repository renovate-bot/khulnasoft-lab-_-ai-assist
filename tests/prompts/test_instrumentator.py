import datetime
from unittest.mock import call, patch

import pytest

from ai_gateway.prompts.instrumentator import PromptInstrumentator


@pytest.fixture
def instrumentator():
    return PromptInstrumentator()


class TestPromptInstrumentator:
    @patch("prometheus_client.Gauge.labels")
    def test_log_pre_api_call(self, mock_gauges, instrumentator):
        instrumentator.log_pre_api_call("claude", [], {"model": "claude"})

        assert mock_gauges.mock_calls == [
            call(model_engine="litellm", model_name="claude"),
            call().inc(),
        ]

    @pytest.mark.asyncio
    @patch("prometheus_client.Gauge.labels")
    @patch("prometheus_client.Counter.labels")
    @patch("prometheus_client.Histogram.labels")
    async def test_async_log_success_event(
        self, mock_histograms, mock_counters, mock_gauges, instrumentator
    ):
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=1)

        await instrumentator.async_log_success_event(
            {"model": "claude"}, None, start_time, end_time
        )

        assert mock_gauges.mock_calls == [
            call(model_engine="litellm", model_name="claude"),
            call().dec(),
        ]

        assert mock_counters.mock_calls == [
            call(
                model_engine="litellm",
                model_name="claude",
                error="no",
                streaming="yes",
                feature_category="unknown",
            ),
            call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            call(
                model_engine="litellm",
                model_name="claude",
                error="no",
                streaming="yes",
                feature_category="unknown",
            ),
            call().observe(1.0),
        ]

    @pytest.mark.asyncio
    @patch("prometheus_client.Gauge.labels")
    @patch("prometheus_client.Counter.labels")
    @patch("prometheus_client.Histogram.labels")
    async def test_async_log_failure_event(
        self, mock_histograms, mock_counters, mock_gauges, instrumentator
    ):
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=1)

        await instrumentator.async_log_failure_event(
            {"model": "claude"}, None, start_time, end_time
        )

        assert mock_gauges.mock_calls == [
            call(model_engine="litellm", model_name="claude"),
            call().dec(),
        ]

        assert mock_counters.mock_calls == [
            call(
                model_engine="litellm",
                model_name="claude",
                error="yes",
                streaming="yes",
                feature_category="unknown",
            ),
            call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            call(
                model_engine="litellm",
                model_name="claude",
                error="yes",
                streaming="yes",
                feature_category="unknown",
            ),
            call().observe(1.0),
        ]
