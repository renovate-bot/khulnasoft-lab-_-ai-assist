import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.tracking.errors import log_exception

__all__ = [
    "TokenAuthority",
]


class TokenAuthority:
    def __init__(self, signing_key):
        self.signing_key = signing_key

    def encode(self, sub) -> tuple[str, int]:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        try:
            claims = {
                "iss": "gitlab-ai-gateway",
                "sub": sub,
                "aud": "gitlab-ai-gateway",
                "exp": expires_at,
                "nbf": datetime.now(timezone.utc),
                "iat": datetime.now(timezone.utc),
                "jti": str(uuid.uuid4()),
                "gitlab_realm": "self-managed",
                "scopes": [GitLabUnitPrimitive.CODE_SUGGESTIONS],
            }

            token = jwt.encode(claims, self.signing_key, algorithm="RS256")

            return (token, int(expires_at.timestamp()))
        except JWTError as err:
            log_exception(err)
            raise
