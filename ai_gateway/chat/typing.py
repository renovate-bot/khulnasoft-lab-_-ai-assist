from typing import Literal

from pydantic import BaseModel

__all__ = ["Resource"]


class Resource(BaseModel, frozen=True):
    type: Literal["issue", "epic"]
    content: str
