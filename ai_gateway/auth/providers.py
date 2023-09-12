import logging
import urllib.parse
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta

import requests
from jose import JWTError, jwt

from ai_gateway.auth.cache import LocalAuthCache
from ai_gateway.auth.user import User, UserClaims

__all__ = [
    "AuthProvider",
    "GitLabOidcProvider",
]

REQUEST_TIMEOUT_SECONDS = 10


class AuthProvider(ABC):
    @abstractmethod
    def authenticate(self, *args, **kwargs) -> User:
        pass


class GitLabOidcProvider(AuthProvider):
    CACHE_KEY = "jwks"
    AUDIENCE = "gitlab-code-suggestions"
    ALGORITHM = "RS256"
    DEFAULT_REALM = "saas"

    def __init__(self, oidc_providers: dict[str, str], expiry_seconds: int = 86400):
        self.oidc_providers = oidc_providers
        self.expiry_seconds = expiry_seconds
        self.cache = LocalAuthCache()

    def authenticate(self, token: str) -> User:
        jwks = self._jwks()

        is_allowed = True
        third_party_ai_features_enabled = False
        gitlab_realm = self.DEFAULT_REALM
        try:
            jwt_claims = jwt.decode(
                token, jwks, audience=self.AUDIENCE, algorithms=[self.ALGORITHM]
            )
            third_party_ai_features_enabled = jwt_claims.get(
                "third_party_ai_features_enabled", False
            )
            gitlab_realm = jwt_claims.get("gitlab_realm", self.DEFAULT_REALM)
        except JWTError as err:
            logging.error(f"Failed to decode JWT token: {err}")
            is_allowed = False

        return User(
            authenticated=is_allowed,
            claims=UserClaims(
                is_third_party_ai_default=third_party_ai_features_enabled,
                gitlab_realm=gitlab_realm,
            ),
        )

    def _jwks(self) -> dict:
        jwks_record = self.cache.get(self.CACHE_KEY)

        if jwks_record and jwks_record.exp > datetime.now():
            return jwks_record.value

        jwks = defaultdict(list)

        for oidc_provider, base_url in self.oidc_providers.items():
            well_known = self._fetch_well_known(oidc_provider, base_url)
            for k, v in self._fetch_jwks(oidc_provider, well_known).items():
                jwks[k].extend(v)

        if jwks:
            self._cache_jwks(jwks)
            return jwks

        return {}

    def _cache_jwks(self, jwks):
        exp = datetime.now() + timedelta(seconds=self.expiry_seconds)
        self.cache.set(self.CACHE_KEY, jwks, exp)

    def _fetch_well_known(self, oidc_provider, base_url: str) -> dict:
        end_point = "/.well-known/openid-configuration"
        url = urllib.parse.urljoin(base_url, end_point)

        well_known = {}
        try:
            res = requests.get(url=url, timeout=REQUEST_TIMEOUT_SECONDS)
            well_known = res.json()
        except requests.exceptions.RequestException as err:
            logging.error(
                f"Unable to fetch OpenID configuration from {oidc_provider}: {err}"
            )

        return well_known

    def _fetch_jwks(self, oidc_provider, well_known) -> dict:
        url = well_known["jwks_uri"]

        jwks = {}

        try:
            res = requests.get(url=url, timeout=REQUEST_TIMEOUT_SECONDS)
            jwks = res.json()
        except requests.exceptions.RequestException as err:
            logging.error(f"Unable to fetch jwks from {oidc_provider}: {err}")

        return jwks
