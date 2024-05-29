import re
import typing

import fastapi

from ai_gateway.auth.gcp import access_token
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.vertex_text import KindVertexTextModel
from ai_gateway.proxy.clients.base import BaseProxyClient


class VertexAIProxyClient(BaseProxyClient):
    def __init__(self, project: str, location: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.location = location

    ALLOWED_HEADERS_TO_UPSTREAM = [
        "content-type",
    ]

    ALLOWED_HEADERS_TO_DOWNSTREAM = ["content-type"]

    def _extract_upstream_path(self, request_path: str) -> str:
        model, action, sse_flag = self._extract_params_from_path(request_path)
        return f"/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{model}:{action}{sse_flag}"

    def _allowed_upstream_paths(self) -> list[str]:
        return []  # No-op. _extract_upstream_path is overridden instead.

    def _allowed_headers_to_upstream(self):
        return VertexAIProxyClient.ALLOWED_HEADERS_TO_UPSTREAM

    def _allowed_headers_to_downstream(self):
        return VertexAIProxyClient.ALLOWED_HEADERS_TO_DOWNSTREAM

    def _allowed_upstream_models(self):
        return [el.value for el in KindVertexTextModel]

    def _upstream_service(self):
        return KindModelProvider.VERTEX_AI.value

    def _extract_model_name(self, upstream_path: str, json_body: typing.Any) -> str:
        model, _, _ = self._extract_params_from_path(upstream_path)
        return model

    def _extract_stream_flag(self, upstream_path: str, json_body: typing.Any) -> bool:
        _, action, _ = self._extract_params_from_path(upstream_path)
        return action == "serverStreamingPredict"

    def _update_headers_to_upstream(self, headers_to_upstream: typing.Any) -> None:
        headers_to_upstream["Authorization"] = f"Bearer {access_token()}"

    def _extract_params_from_path(self, path: str) -> tuple[str, str, str]:
        match = re.search(
            "/v1/projects/.*/locations/.*/publishers/google/models/(.*):(predict|serverStreamingPredict)(\\?alt=sse)?",
            path,
        )

        try:
            assert match is not None

            model = match.group(1)
            action = match.group(2)
            sse_flag = match.group(3) or ""
        except (IndexError, AssertionError):
            raise fastapi.HTTPException(status_code=404, detail="Not found")

        return model, action, sse_flag
