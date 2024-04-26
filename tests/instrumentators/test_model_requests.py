import asyncio
from typing import Awaitable, Callable
from unittest import mock

import pytest

from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator


class TestModelRequestInstrumentator:
    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    def test_watch_sync(
        self, time_counter, mock_histograms, mock_counters, mock_gauges
    ):
        time_counter.side_effect = [1, 2]

        instrumentator = ModelRequestInstrumentator(
            model_engine="anthropic", model_name="claude", concurrency_limit=None
        )
        with instrumentator.watch():
            assert mock_gauges.mock_calls == [
                mock.call(model_engine="anthropic", model_name="claude"),
                mock.call().inc(),
            ]

            mock_gauges.reset_mock()

        assert mock_counters.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
            ),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
            ),
            mock.call().observe(1),
        ]

    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    def test_watch_sync_with_error(
        self, time_counter, mock_histograms, mock_counters, mock_gauges
    ):
        time_counter.side_effect = [1, 2]

        instrumentator = ModelRequestInstrumentator(
            model_engine="anthropic", model_name="claude", concurrency_limit=None
        )

        with pytest.raises(ValueError):
            with instrumentator.watch():
                assert mock_gauges.mock_calls == [
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().inc(),
                ]

                mock_gauges.reset_mock()

                raise ValueError("broken")

        assert mock_gauges.mock_calls == [
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().dec(),
        ]
        assert mock_counters.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="yes",
                streaming="no",
            ),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="yes",
                streaming="no",
            ),
            mock.call().observe(1),
        ]

    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_with_limit(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(
            model_engine="anthropic", model_name="claude", concurrency_limit=5
        )

        with instrumentator.watch():
            mock_gauges.assert_has_calls(
                [
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().set(5),
                ]
            )

    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    def test_watch_async(
        self, time_counter, mock_histograms, mock_counters, mock_gauges
    ):
        time_counter.side_effect = [1, 2]
        instrumentator = ModelRequestInstrumentator(
            model_engine="anthropic", model_name="claude", concurrency_limit=None
        )

        with instrumentator.watch(stream=True) as watcher:
            assert mock_gauges.mock_calls == [
                mock.call(model_engine="anthropic", model_name="claude"),
                mock.call().inc(),
            ]

            mock_gauges.reset_mock()

            watcher.finish()

            assert mock_gauges.mock_calls == [
                mock.call(model_engine="anthropic", model_name="claude"),
                mock.call().dec(),
            ]
            assert mock_counters.mock_calls == [
                mock.call(
                    model_engine="anthropic",
                    model_name="claude",
                    error="no",
                    streaming="yes",
                ),
                mock.call().inc(),
            ]
            assert mock_histograms.mock_calls == [
                mock.call(
                    model_engine="anthropic",
                    model_name="claude",
                    error="no",
                    streaming="yes",
                ),
                mock.call().observe(1),
            ]
