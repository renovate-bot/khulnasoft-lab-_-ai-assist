from pydantic import BaseModel


class SafetyAttributes(BaseModel):
    blocked: bool = False
    categories: list[str] = []
    errors: list[int] = []
