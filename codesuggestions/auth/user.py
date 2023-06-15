from typing import NamedTuple

__all__ = [
    "User",
    "UserClaims",
]


class UserClaims(NamedTuple):
    is_third_party_ai_default: bool
    gitlab_realm: str = 'saas'


class User(NamedTuple):
    authenticated: bool
    claims: UserClaims
