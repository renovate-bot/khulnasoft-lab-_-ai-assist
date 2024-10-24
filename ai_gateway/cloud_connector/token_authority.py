import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from ai_gateway.cloud_connector.config import CloudConnectorConfig
from ai_gateway.cloud_connector.logging import log_exception
from ai_gateway.cloud_connector.providers import CompositeProvider

__all__ = [
    "TokenAuthority",
]


class TokenAuthority:
    ALGORITHM = CompositeProvider.RS256_ALGORITHM

    def __init__(self, signing_key):
        self.signing_key = signing_key

    def encode(
        self, sub, gitlab_realm, current_user, gitlab_instance_id, scopes
    ) -> tuple[str, int]:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        try:
            self_signed_token_issuer = CloudConnectorConfig().service_name
            claims = {
                "iss": self_signed_token_issuer,
                "sub": sub,
                "aud": self_signed_token_issuer,
                "exp": expires_at,
                "nbf": datetime.now(timezone.utc),
                "iat": datetime.now(timezone.utc),
                "jti": str(uuid.uuid4()),
                "gitlab_realm": gitlab_realm,
                "gitlab_instance_id": gitlab_instance_id,
                "scopes": scopes,
            }

            duo_seat_count = getattr(current_user.claims, "duo_seat_count", "")
            if duo_seat_count and str(duo_seat_count).strip():
                claims["duo_seat_count"] = duo_seat_count

            token = jwt.encode(claims, self.signing_key, algorithm=self.ALGORITHM)

            return token, int(expires_at.timestamp())
        except JWTError as err:
            log_exception(err)
            raise
