import datetime
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator


class AgentInstrumentator(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        self._start(kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._stop(kwargs, start_time, end_time, success=True)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._stop(kwargs, start_time, end_time, success=False)

    def _model_instrumentator(self, kwargs):
        return ModelRequestInstrumentator.WatchContainer(
            labels={"model_engine": "litellm", "model_name": kwargs["model"]},
            streaming=True,
            concurrency_limit=None,  # TODO: Plug concurrency limit into agents
        )

    def _start(self, kwargs: dict[str, Any]):
        model_instrumentator = self._model_instrumentator(kwargs)
        model_instrumentator.start()

    def _stop(
        self,
        kwargs: dict[str, Any],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        success: bool,
    ):
        model_instrumentator = self._model_instrumentator(kwargs)

        if not success:
            model_instrumentator.register_error()

        model_instrumentator.finish(duration=(end_time - start_time).total_seconds())
