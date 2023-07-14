import logging
import os
import urllib.parse
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from hashlib import pbkdf2_hmac
from collections import defaultdict

import requests
from jose import JWTError, jwt

from codesuggestions.auth.cache import LocalAuthCache, AuthRecord
from codesuggestions.auth.user import User, UserClaims

__all__ = [
    "AuthProvider",
    "GitLabAuthProvider",
    "GitLabOidcProvider",
]

REQUEST_TIMEOUT_SECONDS = 10


class AuthProvider(ABC):
    @abstractmethod
    def authenticate(self, *args, **kwargs) -> User:
        pass


class GitLabAuthProvider(AuthProvider):
    REQUEST_TIMEOUT_SECONDS = 1

    def __init__(self, base_url: str, expiry_seconds: int = 3600):
        self.base_url = base_url
        self.expiry_seconds = expiry_seconds
        self.cache = LocalAuthCache()
        self.salt = os.urandom(16)

    def _request_code_suggestions_allowed(self, token: str) -> User:
        end_point = "ml/ai-assist"
        url = urllib.parse.urljoin(self.base_url, end_point)
        headers = dict(Authorization=f"Bearer {token}")

        user_authenticated = False
        third_party_enabled = False

        try:
            res = requests.get(
                url=url, headers=headers, timeout=self.REQUEST_TIMEOUT_SECONDS
            )
            if 200 <= res.status_code < 300:
                res_body = res.json()
                user_authenticated = res_body.get("user_is_allowed", False)
                third_party_enabled = res_body.get(
                    "third_party_ai_features_enabled", False
                )
        except requests.exceptions.RequestException as e:
            logging.error(f"Unable to authenticate user with GitLab API: {e}")

        return User(
            authenticated=user_authenticated,
            claims=UserClaims(is_third_party_ai_default=third_party_enabled),
        )

    def _is_auth_required(self, record: AuthRecord) -> bool:
        if record is None:
            return True

        if record.exp <= datetime.now():
            # Key is in cache but it's expired
            return True

        return False

    def _cache_auth(self, key: str, user: User):
        exp = datetime.now() + timedelta(seconds=self.expiry_seconds)
        self.cache.set(key, user, exp)

    def _hash_token(self, token: str) -> str:
        return pbkdf2_hmac("sha256", token.encode(), self.salt, 10_000).hex()

    def authenticate(self, token: str) -> User:
        """
        Checks if the user is allowed to use Code Suggestions
        :param token: Users Personal Access Token or OAuth token
        :return: bool
        """
        key = self._hash_token(token)
        record = self.cache.get(key)
        if not self._is_auth_required(record):
            return record.value

        # authenticate user sending the GitLab API request
        user = self._request_code_suggestions_allowed(token)
        self._cache_auth(key, user)

        return user


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
                gitlab_realm=gitlab_realm
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
            logging.error(f"Unable to fetch OpenID configuration from {oidc_provider}: {err}")

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
