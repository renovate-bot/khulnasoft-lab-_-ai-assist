from typing import Literal

from pydantic import BaseModel

__all__ = ["Context"]


class Context(BaseModel, frozen=True):
    type: Literal["issue", "epic"]
    content: str
