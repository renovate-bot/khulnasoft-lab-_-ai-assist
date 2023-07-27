from abc import ABC, abstractmethod
from datetime import datetime
from typing import NamedTuple, Optional

from codesuggestions.auth.user import User

__all__ = [
    "AuthRecord",
    "BaseAuthCache",
    "LocalAuthCache",
]


class AuthRecord(NamedTuple):
    value: User
    exp: datetime


class BaseAuthCache(ABC):
    def __init__(self, expiry_seconds: int = 3600):
        self.expiry_seconds = expiry_seconds

    @abstractmethod
    def set(self, k: str, val: str, exp: datetime):
        pass

    @abstractmethod
    def get(self, k: str) -> Optional[AuthRecord]:
        pass

    @abstractmethod
    def delete(self, k: str):
        pass


class LocalAuthCache(BaseAuthCache):
    def __init__(self):
        super().__init__()
        self.in_memory_cache = dict()

    def set(self, k: str, val: User, exp: datetime):
        self.in_memory_cache[k] = AuthRecord(
            value=val,
            exp=exp,
        )

    def get(self, k: str) -> Optional[AuthRecord]:
        record = self.in_memory_cache.get(k, None)
        if record is None:
            return None
        return record

    def delete(self, k: str):
        self.in_memory_cache.pop(k, None)
