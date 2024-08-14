from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.code.typing import Token
from ai_gateway.async_dependency_resolver import (
    get_internal_event_client,
    get_token_authority,
)
from ai_gateway.auth.self_signed_jwt import SELF_SIGNED_TOKEN_ISSUER, TokenAuthority
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.internal_events import InternalEventsClient

__all__ = [
    "router",
]


log = structlog.stdlib.get_logger("user_access_token")

router = APIRouter()


@router.post("/user_access_token")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
async def user_access_token(
    request: Request,
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
    token_authority: TokenAuthority = Depends(get_token_authority),
    x_gitlab_global_user_id: Annotated[
        str, Header()
    ] = None,  # This is the value of X_GITLAB_GLOBAL_USER_ID_HEADER
    x_gitlab_realm: Annotated[
        str, Header()
    ] = None,  # This is the value of X_GITLAB_REALM_HEADER
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    if not current_user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=[SELF_SIGNED_TOKEN_ISSUER],
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to create user access token for code suggestions",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.CODE_SUGGESTIONS}",
        category=__name__,
    )

    if not x_gitlab_global_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Gitlab-Global-User-Id header",
        )

    if not x_gitlab_realm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Gitlab-Realm header",
        )

    try:
        token, expires_at = token_authority.encode(
            x_gitlab_global_user_id, x_gitlab_realm, current_user
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate JWT")

    return Token(token=token, expires_at=expires_at)
