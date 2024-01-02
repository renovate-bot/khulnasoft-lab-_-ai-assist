import urllib.parse
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta

import requests
from jose import JWTError, jwt

from ai_gateway.auth.cache import LocalAuthCache
from ai_gateway.auth.user import User, UserClaims
from ai_gateway.tracking.errors import log_exception

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
    ALGORITHM = "RS256"
    DEFAULT_REALM = "saas"
    AUDIENCE = "gitlab-ai-gateway"

    class CriticalAuthError(Exception):
        pass

    def __init__(self, oidc_providers: dict[str, str], expiry_seconds: int = 86400):
        self.oidc_providers = oidc_providers
        self.expiry_seconds = expiry_seconds
        self.cache = LocalAuthCache()

    def authenticate(self, token: str) -> User:
        jwks = self._jwks()
        if len(jwks.get("keys", [])) == 0:
            raise self.CriticalAuthError(
                "No keys founds in JWKS; are OIDC providers up?"
            )

        is_allowed = False
        gitlab_realm = self.DEFAULT_REALM
        scopes = []

        try:
            jwt_claims = jwt.decode(
                token, jwks, audience=self.AUDIENCE, algorithms=[self.ALGORITHM]
            )
            gitlab_realm = jwt_claims.get("gitlab_realm", self.DEFAULT_REALM)
            scopes = jwt_claims.get("scopes", [])
            is_allowed = True
        except JWTError as err:
            log_exception(err)

        return User(
            authenticated=is_allowed,
            claims=UserClaims(
                gitlab_realm=gitlab_realm,
                scopes=scopes,
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
            log_exception(err, {"oidc_provider": oidc_provider})

        return well_known

    def _fetch_jwks(self, oidc_provider, well_known) -> dict:
        url = well_known.get("jwks_uri")
        if not url:
            return {}

        jwks = {}

        try:
            res = requests.get(url=url, timeout=REQUEST_TIMEOUT_SECONDS)
            jwks = res.json()
        except requests.exceptions.RequestException as err:
            log_exception(err, {"oidc_provider": oidc_provider})

        return jwks
