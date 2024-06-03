from typing import NamedTuple, Optional

from fastapi import Request
from starlette.authentication import BaseUser

from ai_gateway.gitlab_features import GitLabUnitPrimitive

__all__ = ["User", "UserClaims", "GitLabUser"]


class UserClaims(NamedTuple):
    gitlab_realm: str = ""
    scopes: list = []
    subject: str = ""
    issuer: str = ""


class User(NamedTuple):
    authenticated: bool
    claims: UserClaims


class GitLabUser(BaseUser):
    def __init__(
        self,
        authenticated: bool,
        is_debug: bool = False,
        claims: Optional[UserClaims] = None,
    ):
        self._authenticated = authenticated
        self._is_debug = is_debug
        self._claims = claims

    @property
    def claims(self) -> Optional[UserClaims]:
        return self._claims

    @property
    def is_debug(self) -> bool:
        return self._is_debug

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    def can(
        self, unit_primitive: GitLabUnitPrimitive, disallowed_issuers: list[str] = None
    ) -> bool:
        if not unit_primitive:
            return False

        if self.is_debug:
            return True

        if disallowed_issuers and self._claims.issuer in disallowed_issuers:
            return False

        return unit_primitive in self._claims.scopes


async def get_current_user(request: Request) -> GitLabUser:
    return request.user
