import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.tracking.errors import log_exception

__all__ = [
    "TokenAuthority",
]


SELF_SIGNED_TOKEN_ISSUER = "gitlab-ai-gateway"


class TokenAuthority:
    def __init__(self, signing_key):
        self.signing_key = signing_key

    def encode(self, sub, gitlab_realm) -> tuple[str, int]:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        try:
            claims = {
                "iss": SELF_SIGNED_TOKEN_ISSUER,
                "sub": sub,
                "aud": SELF_SIGNED_TOKEN_ISSUER,
                "exp": expires_at,
                "nbf": datetime.now(timezone.utc),
                "iat": datetime.now(timezone.utc),
                "jti": str(uuid.uuid4()),
                "gitlab_realm": gitlab_realm,
                "scopes": [GitLabUnitPrimitive.CODE_SUGGESTIONS],
            }

            token = jwt.encode(claims, self.signing_key, algorithm="RS256")

            return (token, int(expires_at.timestamp()))
        except JWTError as err:
            log_exception(err)
            raise
