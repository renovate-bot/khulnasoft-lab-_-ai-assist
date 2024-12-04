import base64
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone

import requests
from gitlab_cloud_connector import CompositeProvider
from jose import jwk, jwt


class GlgoAuthority:
    ALGORITHM = CompositeProvider.RS256_ALGORITHM
    AUDIENCE = "glgo"
    ISSUER = "https://cloud.gitlab.com"
    JWT = "JWT"

    def __init__(
        self,
        signing_key: str,
        glgo_base_url: str,
    ):
        self.signing_key = signing_key
        self.kid = self._build_kid(signing_key)
        self.token_endpoint = f"{glgo_base_url}/cc/token"

    def token(self, user_id: str, cloud_connector_token: str):
        token = self._build_token(user_id, cloud_connector_token)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        data = {"user_id": user_id}
        response = requests.post(
            url=self.token_endpoint,
            json=data,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        return response.json().get("token")

    def _build_token(self, user_id: str, cloud_connector_token: str):
        now = datetime.now(timezone.utc)
        payload = {
            "iss": self.ISSUER,
            "aud": self.AUDIENCE,
            "exp": now + timedelta(hours=1),
            "nbf": now,
            "iat": now,
            "jti": str(uuid.uuid4()),
            "cct": cloud_connector_token,
            "uid": user_id,
        }

        token = jwt.encode(
            payload,
            self.signing_key,
            algorithm=self.ALGORITHM,
            headers={"typ": self.JWT, "kid": self.kid},
        )

        return token

    def _build_kid(self, signing_key):
        jwk_dict = (
            jwk.RSAKey(algorithm=self.ALGORITHM, key=signing_key).public_key().to_dict()
        )
        normalized_data = {
            key: jwk_dict[key] for key in ["e", "kty", "n"] if key in jwk_dict
        }
        serialized_data = json.dumps(
            normalized_data, separators=(",", ":"), sort_keys=True
        ).encode()
        sha256_hash = hashlib.sha256(serialized_data).digest()

        return base64.urlsafe_b64encode(sha256_hash).decode().removesuffix("=")
