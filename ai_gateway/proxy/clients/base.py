import json
import re
import typing
from abc import ABC, abstractmethod

import fastapi
import httpx
from starlette.background import BackgroundTask

from ai_gateway.config import ConfigModelConcurrency
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator


class BaseProxyClient(ABC):
    def __init__(
        self, client: httpx.AsyncClient, concurrency_limit: ConfigModelConcurrency
    ):
        self.client = client
        self.concurrency_limit = concurrency_limit

    async def proxy(self, request: fastapi.Request) -> fastapi.Response:
        upstream_path = self._extract_upstream_path(request.url.path)
        json_body = await self._extract_json_body(request)
        model_name = self._extract_model_name(json_body)

        if model_name not in self._allowed_upstream_models():
            raise fastapi.HTTPException(status_code=400, detail="Unsupported model")

        stream = self._extract_stream_flag(json_body)
        headers_to_upstream = self._create_headers_to_upstream(request.headers)
        self._update_headers_to_upstream(headers_to_upstream)

        request_to_upstream = self.client.build_request(
            request.method,
            httpx.URL(path=upstream_path),
            headers=headers_to_upstream,
            json=json_body,
        )

        with ModelRequestInstrumentator(
            model_engine=self._upstream_service(),
            model_name=model_name,
            concurrency_limit=self.concurrency_limit.for_model(
                engine=self._upstream_service(), name=model_name
            ),
        ).watch(stream=stream) as watcher:
            try:
                response_from_upstream = await self.client.send(
                    request_to_upstream, stream=stream
                )
            except Exception:
                watcher.register_error()
                watcher.finish()
                raise fastapi.HTTPException(status_code=502, detail="Bad Gateway")

        headers_to_downstream = self._create_headers_to_downstream(
            response_from_upstream.headers
        )

        if stream:
            return fastapi.responses.StreamingResponse(
                response_from_upstream.aiter_text(),
                status_code=response_from_upstream.status_code,
                headers=headers_to_downstream,
                background=BackgroundTask(func=watcher.afinish),
            )

        return fastapi.Response(
            content=response_from_upstream.content,
            status_code=response_from_upstream.status_code,
            headers=headers_to_downstream,
        )

    @abstractmethod
    def _allowed_upstream_paths(self) -> list[str]:
        """Allowed paths to the upstream service."""
        pass

    @abstractmethod
    def _allowed_headers_to_upstream(self) -> list[str]:
        """Allowed request headers to the upstream service."""
        pass

    @abstractmethod
    def _allowed_headers_to_downstream(self) -> list[str]:
        """Allowed response headers to the downstream service."""
        pass

    @abstractmethod
    def _upstream_service(self) -> str:
        """Name of the upstream service."""
        pass

    @abstractmethod
    def _allowed_upstream_models(self) -> list[str]:
        """Allowed models to the upstream service."""
        pass

    @abstractmethod
    def _extract_model_name(self, json_body: typing.Any) -> str:
        """Extract model name from the request."""
        pass

    @abstractmethod
    def _extract_stream_flag(self, json_body: typing.Any) -> bool:
        """Extract stream flag from the request."""
        pass

    @abstractmethod
    def _update_headers_to_upstream(self, headers: dict[str, str]) -> None:
        """Update headers for vendor specific requirements."""
        pass

    def _extract_upstream_path(self, request_path: str) -> str:
        path = re.sub(f"^(.*?)/{self._upstream_service()}/", "/", request_path)

        if path not in self._allowed_upstream_paths():
            raise fastapi.HTTPException(status_code=404, detail="Not found")

        return path

    async def _extract_json_body(self, request: fastapi.Request) -> typing.Any:
        body = await request.body()

        try:
            json_body = json.loads(body)
        except json.JSONDecodeError:
            raise fastapi.HTTPException(status_code=400, detail="Invalid JSON")

        return json_body

    def _create_headers_to_upstream(self, headers_from_downstream) -> dict[str, str]:
        return {
            key: headers_from_downstream[key]
            for key in self._allowed_headers_to_upstream()
            if key in headers_from_downstream
        }

    def _create_headers_to_downstream(self, headers_from_upstream) -> dict[str, str]:
        return {
            key: headers_from_upstream.get(key)
            for key in self._allowed_headers_to_downstream()
            if key in headers_from_upstream
        }
