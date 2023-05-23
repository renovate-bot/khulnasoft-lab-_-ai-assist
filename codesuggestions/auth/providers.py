import logging
import os
import urllib.parse
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from hashlib import pbkdf2_hmac

import requests
from jose import JWTError, jwt

from codesuggestions.auth.cache import LocalAuthCache

__all__ = [
    "AuthProvider",
    "GitLabAuthProvider",
    "GitLabOidcProvider",
]

REQUEST_TIMEOUT_SECONDS = 10


class AuthProvider(ABC):
    @abstractmethod
    def authenticate(self, *args, **kwargs):
        pass


class GitLabAuthProvider(AuthProvider):
    REQUEST_TIMEOUT_SECONDS = 1

    def __init__(self, base_url: str, expiry_seconds: int = 3600):
        self.base_url = base_url
        self.expiry_seconds = expiry_seconds
        self.cache = LocalAuthCache()
        self.salt = os.urandom(16)

    def _request_code_suggestions_allowed(self, token: str) -> bool:
        end_point = "ml/ai-assist"
        url = urllib.parse.urljoin(self.base_url, end_point)
        headers = dict(Authorization=f"Bearer {token}")

        is_allowed = False

        try:
            res = requests.get(url=url, headers=headers, timeout=self.REQUEST_TIMEOUT_SECONDS)
            if not (200 <= res.status_code < 300):
                is_allowed = False
            else:
                is_allowed = res.json().get("user_is_allowed", False)
        except requests.exceptions.RequestException as e:
            logging.error(f"Unable to authenticate user with GitLab API: {e}")

        return is_allowed

    def _is_auth_required(self, key: str) -> bool:
        record = self.cache.get(key)
        if record is None:
            return True

        if record.exp <= datetime.now():
            # Key is in cache but it's expired
            return True

        return False

    def _cache_auth(self, key: str, token: str):
        exp = datetime.now() + timedelta(seconds=self.expiry_seconds)
        self.cache.set(key, token, exp)

    def _hash_token(self, token: str) -> str:
        return pbkdf2_hmac("sha256", token.encode(), self.salt, 10_000).hex()

    def authenticate(self, token: str) -> bool:
        """
        Checks if the user is allowed to use Code Suggestions
        :param token: Users Personal Access Token or OAuth token
        :return: bool
        """
        key = self._hash_token(token)
        if not self._is_auth_required(key):
            return True

        # authenticate user sending the GitLab API request
        is_allowed = self._request_code_suggestions_allowed(token)
        if is_allowed:
            self._cache_auth(key, token)

        return is_allowed


class GitLabOidcProvider(AuthProvider):
    CACHE_KEY = "jwks"
    AUDIENCE = "gitlab-code-suggestions"
    ALGORITHM = "RS256"

    def __init__(self, base_url: str, expiry_seconds: int = 86400):
        self.base_url = base_url
        self.expiry_seconds = expiry_seconds
        self.cache = LocalAuthCache()

    def authenticate(self, token: str) -> bool:
        jwks = self._jwks()

        is_allowed = True
        try:
            _ = jwt.decode(
                token, jwks, audience=self.AUDIENCE, algorithms=[self.ALGORITHM]
            )
        except JWTError as err:
            logging.error(f"Failed to decode JWT token: {err}")
            is_allowed = False

        return is_allowed

    def _jwks(self) -> dict:
        jwks_record = self.cache.get(self.CACHE_KEY)

        if jwks_record and jwks_record.exp > datetime.now():
            return jwks_record.value

        well_known = self._fetch_well_known()
        jwks = self._fetch_jwks(well_known)

        if jwks:
            self._cache_jwks(jwks)
            return jwks

        return {}

    def _cache_jwks(self, jwks):
        exp = datetime.now() + timedelta(seconds=self.expiry_seconds)
        self.cache.set(self.CACHE_KEY, jwks, exp)

    def _fetch_well_known(self) -> dict:
        end_point = "/.well-known/openid-configuration"
        url = urllib.parse.urljoin(self.base_url, end_point)

        well_known = {}
        try:
            res = requests.get(url=url, timeout=REQUEST_TIMEOUT_SECONDS)
            well_known = res.json()
        except requests.exceptions.RequestException as err:
            logging.error(f"Unable to fetch OpenID configuration from GitLab: {err}")

        return well_known

    def _fetch_jwks(self, well_known) -> dict:
        url = well_known["jwks_uri"]

        jwks = {}

        try:
            res = requests.get(url=url, timeout=REQUEST_TIMEOUT_SECONDS)
            jwks = res.json()
        except requests.exceptions.RequestException as err:
            logging.error(f"Unable to fetch jwks from GitLab: {err}")

        return jwks
