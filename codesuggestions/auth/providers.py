import os
import logging
import urllib.parse
from hashlib import pbkdf2_hmac
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import requests

from codesuggestions.auth.cache import LocalAuthCache

__all__ = [
    "AuthProvider",
    "GitLabAuthProvider",
]


class AuthProvider(ABC):
    @abstractmethod
    def authenticate(self, *args, **kwargs):
        pass


class GitLabAuthProvider(AuthProvider):
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
            res = requests.get(url=url, headers=headers)
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
        return (
            pbkdf2_hmac('sha256', token.encode(), self.salt, 10_000)
            .hex()
        )

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
