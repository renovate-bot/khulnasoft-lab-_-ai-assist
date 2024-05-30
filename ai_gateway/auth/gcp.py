from functools import lru_cache

import google.auth
import google.auth.transport.requests
from google.auth.credentials import Credentials, TokenState


def access_token() -> str:
    """
    Get access token from Google Application Default Credentials (ADC).
    See https://google-auth.readthedocs.io/en/latest/user-guide.html#application-default-credentials
    """

    creds = _fetch_application_default_credentials()

    if creds.token_state is not TokenState.FRESH:
        _fetch_application_default_credentials.cache_clear()
        creds = _fetch_application_default_credentials()

    return creds.token


@lru_cache(maxsize=1)
def _fetch_application_default_credentials() -> Credentials:
    creds, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds
