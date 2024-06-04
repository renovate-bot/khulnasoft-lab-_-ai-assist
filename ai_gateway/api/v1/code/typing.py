from pydantic import BaseModel

__all__ = [
    "Token",
]


class Token(BaseModel):
    token: str
    expires_at: int
