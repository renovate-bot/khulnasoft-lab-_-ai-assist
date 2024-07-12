from unittest import mock

import pytest

from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator


class TestWatchContainer:
    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    @pytest.mark.parametrize(
        ("mock_timestamps", "start", "duration", "expected_duration"),
        [([], False, 1.5, 1.5), ([1, 2], True, None, 1), ([], False, None, None)],
    )
    def test_finish(
        self,
        time_counter,
        mock_histograms,
        mock_counters,
        mock_gauges,
        mock_timestamps,
        start,
        duration,
        expected_duration,
    ):
        container = ModelRequestInstrumentator.WatchContainer(
            labels={"model_engine": "anthropic", "model_name": "claude"},
            streaming=False,
            concurrency_limit=None,
        )
        time_counter.side_effect = mock_timestamps

        if start:
            container.start()
            mock_gauges.reset_mock()  # So we only have the calls from `stop` bellow

        if expected_duration:
            container.finish(duration)

            assert mock_gauges.mock_calls == [
                mock.call(model_engine="anthropic", model_name="claude"),
                mock.call().dec(),
            ]

            assert mock_counters.mock_calls == [
                mock.call(
                    model_engine="anthropic",
                    model_name="claude",
                    error="no",
                    streaming="no",
                    feature_category="unknown",
                ),
                mock.call().inc(),
            ]
            assert mock_histograms.mock_calls == [
                mock.call(
                    model_engine="anthropic",
                    model_name="claude",
                    error="no",
                    streaming="no",
                    feature_category="unknown",
                ),
                mock.call().observe(expected_duration),
            ]
        else:
            with pytest.raises(
                ValueError,
                match="start\(\) must be called before finish\(\) if duration is not provided",
            ):
                container.finish(duration)


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
                feature_category="unknown",
            ),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
                feature_category="unknown",
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
                feature_category="unknown",
            ),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="yes",
                streaming="no",
                feature_category="unknown",
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
                    feature_category="unknown",
                ),
                mock.call().inc(),
            ]
            assert mock_histograms.mock_calls == [
                mock.call(
                    model_engine="anthropic",
                    model_name="claude",
                    error="no",
                    streaming="yes",
                    feature_category="unknown",
                ),
                mock.call().observe(1),
            ]
