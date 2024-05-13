import os
import typing

import fastapi

from ai_gateway.models.anthropic import KindAnthropicModel
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients.base import BaseProxyClient


class AnthropicProxyClient(BaseProxyClient):
    ALLOWED_UPSTREAM_PATHS = [
        "/v1/complete",
        "/v1/messages",
    ]

    ALLOWED_HEADERS_TO_UPSTREAM = [
        "accept",
        "content-type",
        "anthropic-version",
    ]

    ALLOWED_HEADERS_TO_DOWNSTREAM = [
        "date",
        "content-type",
        "transfer-encoding",
    ]

    def _allowed_upstream_paths(self) -> list[str]:
        return AnthropicProxyClient.ALLOWED_UPSTREAM_PATHS

    def _allowed_headers_to_upstream(self):
        return AnthropicProxyClient.ALLOWED_HEADERS_TO_UPSTREAM

    def _allowed_headers_to_downstream(self):
        return AnthropicProxyClient.ALLOWED_HEADERS_TO_DOWNSTREAM

    def _allowed_upstream_models(self):
        return [el.value for el in KindAnthropicModel]

    def _upstream_service(self):
        return KindModelProvider.ANTHROPIC.value

    def _extract_model_name(self, upstream_path: str, json_body: typing.Any) -> str:
        try:
            return json_body["model"]
        except KeyError:
            raise fastapi.HTTPException(
                status_code=400, detail="Failed to extract model name"
            )

    def _extract_stream_flag(self, upstream_path: str, json_body: typing.Any) -> bool:
        return json_body.get("stream", False)

    def _update_headers_to_upstream(self, headers_to_upstream: typing.Any) -> None:
        try:
            headers_to_upstream["x-api-key"] = os.environ["ANTHROPIC_API_KEY"]
        except KeyError:
            raise fastapi.HTTPException(status_code=400, detail="API key not found")
