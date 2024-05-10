import time
from functools import lru_cache

import google.auth
import google.auth.transport.requests

TOKEN_TTL = 3540  # 59 minutes in seconds


def access_token() -> str:
    """
    Get access token from Google Application Default Credentials.
    See https://google-auth.readthedocs.io/en/latest/user-guide.html#application-default-credentials
    """

    token, created_at = _fetch_access_token_from_adr()

    if time.time() - created_at > TOKEN_TTL:
        _fetch_access_token_from_adr.cache_clear()
        token, _ = _fetch_access_token_from_adr()

    return token


@lru_cache(maxsize=1)
def _fetch_access_token_from_adr() -> tuple[str, float]:
    creds, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return (creds.token, time.time())
