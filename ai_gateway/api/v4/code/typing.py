import json
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

__all__ = [
    "StreamDelta",
    "StreamSuggestionChunk",
    "StreamEvent",
    "StreamSSEMessage",
]


class StreamDelta(BaseModel):
    content: str


class StreamSuggestionChunk(BaseModel):
    class Choice(BaseModel):
        delta: StreamDelta
        index: int = 0

    choices: list[Choice]


class StreamEvent(StrEnum):
    START = "stream_start"
    END = "stream_end"
    CONTENT_CHUNK = "content_chunk"


class StreamSSEMessage(BaseModel):
    # The order of these keys matter; client expects `event:` to be outputted first
    event: StreamEvent
    data: Optional[dict] = None

    def dump_with_json_data(self) -> dict:
        model_dump = self.model_dump()
        model_dump["data"] = json.dumps(self.data)
        return model_dump
