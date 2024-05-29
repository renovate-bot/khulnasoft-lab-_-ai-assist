from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import (
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_REALM_HEADER,
)
from ai_gateway.async_dependency_resolver import get_token_authority
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.self_signed_jwt import TokenAuthority
from ai_gateway.self_signed_token.token_authority import SELF_SIGNED_TOKEN_ISSUER

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
):
    if not current_user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=[SELF_SIGNED_TOKEN_ISSUER],
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code suggestions",
        )

    gitlab_user_id = request.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER)
    if not gitlab_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Gitlab-Global-User-Id header",
        )

    gitlab_realm = request.headers.get(X_GITLAB_REALM_HEADER)
    if not gitlab_realm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Gitlab-Realm header",
        )

    try:
        token = token_authority.encode(
            request.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER)
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate JWT")

    return {"token": token}
