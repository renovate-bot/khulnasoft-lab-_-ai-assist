import os
import uuid
from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from jose import JWTError, jwt

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import (
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_REALM_HEADER,
)
from ai_gateway.auth.authentication import requires
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive
from ai_gateway.tracking.errors import log_exception

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("user_access_token")

router = APIRouter()


@router.post("/user_access_token")
@requires(GitLabUnitPrimitive.CODE_SUGGESTIONS)
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
async def user_access_token(
    request: Request,
):
    try:
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

        claims = {
            "iss": "gitlab-ai-gateway",
            "sub": gitlab_user_id,
            "aud": "gitlab-ai-gateway",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "nbf": datetime.now(timezone.utc),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4()),
            "gitlab_realm": gitlab_realm,
            "scopes": [GitLabUnitPrimitive.CODE_SUGGESTIONS],
        }

        # pylint: disable=direct-environment-variable-reference
        token = jwt.encode(claims, os.environ["JWT_SIGNING_KEY"], algorithm="RS256")
        # pylint: enable=direct-environment-variable-reference

        return {"token": token}
    except JWTError as err:
        log_exception(err)
