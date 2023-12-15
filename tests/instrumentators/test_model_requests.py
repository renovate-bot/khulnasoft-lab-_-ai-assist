from unittest import mock
from typing import Awaitable, Callable

import pytest
import asyncio

from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator

class AsyncThing:
    def __init__(self, max_iterations: int, operation: Callable[[int], Awaitable[None]] = lambda _: asyncio.sleep(0.01)):
        self.counter = 0
        self.max_iterations = max_iterations
        self.operation = operation

    def __aiter__(self):
        return self

    async def __anext__(self) -> int:
        if self.counter >= self.max_iterations:
            raise StopAsyncIteration

        self.counter += 1
        await self.operation(self.counter)

        return self.counter

class TestModelRequestInstrumeentator:
    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_sync(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(model_engine="anthropic", model_name="claude")

        with pytest.raises(ValueError):
            with instrumentator.watch():
                mock_gauges.assert_has_calls([
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().inc(),
                ])
                raise ValueError("broken")

        mock_gauges.assert_has_calls([
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().inc(),
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().dec(),
        ])


    @mock.patch("prometheus_client.Gauge.labels")
    @pytest.mark.asyncio
    async def test_watch_async(self, mock_gauges):
        results = []

        instrumentator = ModelRequestInstrumentator(model_engine="anthropic", model_name="claude")
        async_thing = AsyncThing(3)

        with instrumentator.watch() as watcher:
            mock_gauges.assert_has_calls([
                mock.call(model_engine="anthropic", model_name="claude"),
                mock.call().inc(),
            ])

            async for r in watcher.handle_and_finish_async(async_thing):
                # Make sure we haven't counted the operation as finished yet
                assert mock_gauges.call_count == 1
                results.append(r)

        assert results == [1, 2, 3]
        mock_gauges.assert_has_calls([
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().inc(),
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().dec(),
        ])

    @mock.patch("prometheus_client.Gauge.labels")
    @pytest.mark.asyncio
    async def test_watch_async_errors(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(model_engine="anthropic", model_name="claude")
        async def operation(i):
            if i > 1:
                raise ValueError("broken iteration")

        async_thing = AsyncThing(3, operation=operation)
        result = []
        with pytest.raises(ValueError):
            with instrumentator.watch() as watcher:
                mock_gauges.assert_has_calls([
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().inc(),
                ])

                async for r in watcher.handle_and_finish_async(async_thing):
                    result.append(r)

        assert result == [1]
        mock_gauges.assert_has_calls([
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().inc(),
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().dec(),
        ])

    @mock.patch("prometheus_client.Gauge.labels")
    @pytest.mark.asyncio
    async def test_watch_async_error_handling(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(model_engine="anthropic", model_name="claude")

        async_thing = AsyncThing(3)
        result = []
        with pytest.raises(ValueError):
            with instrumentator.watch() as watcher:
                mock_gauges.assert_has_calls([
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().inc(),
                ])

                async for r in watcher.handle_and_finish_async(async_thing):
                    if r > 1:
                        raise ValueError("broken handler")
                    result.append(r)

        assert result == [1]
        mock_gauges.assert_has_calls([
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().inc(),
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().dec(),
        ])
