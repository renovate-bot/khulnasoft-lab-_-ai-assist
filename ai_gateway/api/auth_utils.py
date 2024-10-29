from typing import Optional

from fastapi import Request
from starlette.authentication import BaseUser

from ai_gateway.cloud_connector import (
    CloudConnectorUser,
    GitLabUnitPrimitive,
    UserClaims,
)


class StarletteUser(BaseUser):
    def __init__(
        self,
        cloud_connector_user: CloudConnectorUser,
    ):
        self.cloud_connector_user = cloud_connector_user

    # overriding starlette BaseUser methods
    @property
    def is_authenticated(self) -> bool:
        return self.cloud_connector_user.is_authenticated

    @property
    def global_user_id(self) -> str | None:
        return self.cloud_connector_user.global_user_id

    @property
    def claims(self) -> Optional[UserClaims]:
        return self.cloud_connector_user.claims

    @property
    def is_debug(self) -> bool:
        return self.cloud_connector_user.is_debug

    @property
    def unit_primitives(self) -> list[GitLabUnitPrimitive]:
        return self.cloud_connector_user.unit_primitives

    def can(
        self,
        unit_primitive: GitLabUnitPrimitive,
        disallowed_issuers: Optional[list[str]] = None,
    ) -> bool:
        return self.cloud_connector_user.can(
            unit_primitive,
            disallowed_issuers,
        )


async def get_current_user(request: Request) -> StarletteUser:
    return request.user
