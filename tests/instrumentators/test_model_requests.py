import asyncio
from typing import Awaitable, Callable
from unittest import mock

import pytest

from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator


class TestModelRequestInstrumentator:
    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_sync(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(
            model_engine="anthropic", model_name="claude"
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

    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_async(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(
            model_engine="anthropic", model_name="claude"
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
