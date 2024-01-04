from typing import NamedTuple

__all__ = [
    "User",
    "UserClaims",
]


class UserClaims(NamedTuple):
    gitlab_realm: str = ""
    scopes: list = []


class User(NamedTuple):
    authenticated: bool
    claims: UserClaims
