from typing import NamedTuple, Optional

from ai_gateway.cloud_connector.gitlab_features import GitLabUnitPrimitive

__all__ = ["UserClaims", "CloudConnectorUser"]


class UserClaims(NamedTuple):
    gitlab_realm: str = ""
    scopes: list = []
    subject: str = ""
    issuer: str = ""
    duo_seat_count: str = ""
    gitlab_instance_id: str = ""


class CloudConnectorUser:
    def __init__(
        self,
        authenticated: bool,
        is_debug: bool = False,
        global_user_id: Optional[str] = None,
        claims: Optional[UserClaims] = None,
    ):
        self._authenticated = authenticated
        self._is_debug = is_debug
        self._claims = claims
        self._global_user_id = global_user_id

    @property
    def global_user_id(self) -> str | None:
        return self._global_user_id

    @property
    def claims(self) -> Optional[UserClaims]:
        return self._claims

    @property
    def is_debug(self) -> bool:
        return self._is_debug

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    @property
    def authenticated(self) -> bool:
        return self.is_authenticated

    @property
    def unit_primitives(self) -> list[GitLabUnitPrimitive]:
        unit_primitives = []

        if not self.claims:
            return []

        for scope in self.claims.scopes:
            try:
                unit_primitives.append(GitLabUnitPrimitive(scope))
            except ValueError:
                pass

        return unit_primitives

    def can(
        self,
        unit_primitive: GitLabUnitPrimitive,
        disallowed_issuers: Optional[list[str]] = None,
    ) -> bool:
        if not unit_primitive:
            return False

        if self.is_debug:
            return True

        if not self._claims:
            return False

        if disallowed_issuers and self._claims.issuer in disallowed_issuers:
            return False

        return unit_primitive in self._claims.scopes
