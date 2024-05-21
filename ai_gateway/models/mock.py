import json
import re
from typing import Any, AsyncIterator, Callable, Optional, TypeVar
from unittest.mock import AsyncMock

import fastapi
import httpx
from anthropic.types import Message

from ai_gateway.models.base import (
    ModelMetadata,
    SafetyAttributes,
    TextGenBaseModel,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.models.base_chat import ChatModelBase

__all__ = [
    "AsyncStream",
    "AsyncClient",
    "LLM",
    "ChatModel",
]

_T = TypeVar("_T")


class AsyncStream(AsyncIterator[_T]):
    def __init__(self, chunks: list[_T], callback_finish: Optional[Callable] = None):
        self.chunks = chunks
        self.callback_finish = callback_finish

    def __aiter__(self) -> "AsyncStream[_T]":
        return self

    async def __anext__(self) -> _T:
        if len(self.chunks) > 0:
            return self.chunks.pop(0)

        if self.callback_finish:
            self.callback_finish()

        raise StopAsyncIteration


class AsyncClient(AsyncMock):
    async def send(self, *args, **kwargs):
        return httpx.Response(
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "date": "2024",
                "transfer-encoding": "chunked",
            },
            json={"response": "mocked"},
        )


class ProxyClient(AsyncMock):
    async def proxy(self, *args, **kwargs):
        return fastapi.Response(
            content=json.dumps({"response": "mocked"}).encode("utf-8"),
            status_code=200,
            headers={"Content-Type": "application/json"},
        )


class SearchClient(AsyncMock):
    async def search(self, *args, **kwargs):
        return {}


class LLM(TextGenBaseModel):
    """
    Implementation of the stub model that inherits the `TextGenBaseModel` interface.
    Please, use this class if you require to mock such models as `AnthropicModel` or `PalmCodeGeckoModel`
    """

    def __init__(self, *_args: Any, **_kwargs: Any):
        super().__init__()

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(name="llm-mocked", engine="llm-provider-mocked")

    async def generate(
        self,
        prefix: str,
        suffix: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> TextGenModelOutput | AsyncIterator[TextGenModelChunk]:
        scope = {
            "prefix": prefix,
            "suffix": suffix,
            "stream": stream,
            "kwargs": dict(kwargs),
        }
        suggestion = (
            f"echo: {json.dumps(scope)}"  # echo the current scope's local variables
        )

        with self.instrumentator.watch(stream=stream) as watcher:
            if stream:
                chunks = [
                    TextGenModelChunk(text=chunk)
                    for chunk in re.split(r"(\s)", suggestion)
                ]
                return AsyncStream(chunks, lambda: watcher.finish())

        return TextGenModelOutput(
            text=suggestion,
            score=0,
            safety_attributes=SafetyAttributes(),
        )


class ChatModel(ChatModelBase):
    """
    Implementation of the stub model that inherits the `ChatModelBase` interface.
    Please, use this class if you require to mock such models as `AnthropicChatModel`
    """

    def __init__(self, *_args: Any, **_kwargs: Any):
        super().__init__()

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="chat-model-mocked", engine="chat-model-provider-mocked"
        )

    async def generate(
        self,
        messages: list[Message],
        stream: bool = False,
        **kwargs: Any,
    ) -> TextGenModelOutput | AsyncIterator[TextGenModelChunk]:
        messages = [message.model_dump(mode="json") for message in messages]

        scope = {"messages": messages, "stream": stream, "kwargs": dict(kwargs)}
        suggestion = (
            f"echo: {json.dumps(scope)}"  # echo the current scope's local variables
        )

        with self.instrumentator.watch(stream=stream) as watcher:
            if stream:
                chunks = [
                    TextGenModelChunk(text=chunk)
                    for chunk in re.split(r"(\s)", suggestion)
                ]
                return AsyncStream(chunks, lambda: watcher.finish())

        return TextGenModelOutput(
            text=suggestion,
            score=0,
            safety_attributes=SafetyAttributes(),
        )
