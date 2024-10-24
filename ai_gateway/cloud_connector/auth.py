from typing import Dict, Optional, Tuple

from ai_gateway.cloud_connector.config import CloudConnectorConfig
from ai_gateway.cloud_connector.logging import log_exception
from ai_gateway.cloud_connector.providers import AuthProvider
from ai_gateway.cloud_connector.user import CloudConnectorUser, UserClaims
from ai_gateway.cloud_connector.validators import (
    X_GITLAB_DUO_SEAT_COUNT_HEADER,
    validate_duo_seat_count_header,
)


class CloudConnectorAuthError:
    def __init__(self, error_message: str):
        self._error_message = error_message

    @property
    def error_message(self) -> str:
        return self._error_message


PREFIX_BEARER_HEADER = "bearer"
AUTH_HEADER = "Authorization"
AUTH_TYPE_HEADER = "X-Gitlab-Authentication-Type"
OIDC_AUTH = "oidc"
X_GITLAB_GLOBAL_USER_ID_HEADER = "X-Gitlab-Global-User-Id"
X_GITLAB_REALM_HEADER = "X-Gitlab-Realm"
X_GITLAB_INSTANCE_ID_HEADER = "X-Gitlab-Instance-Id"
X_GITLAB_HOST_NAME_HEADER = "X-Gitlab-Host-Name"
X_GITLAB_VERSION_HEADER = "X-Gitlab-Version"


def authenticate(
    headers: Dict[str, str],
    oidc_auth_provider: Optional[AuthProvider],
    bypass_auth: bool = False,
) -> Tuple[CloudConnectorUser, Optional[CloudConnectorAuthError]]:
    headers = {k.lower(): v for k, v in headers.items()}
    global_user_id = headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER.lower())

    if bypass_auth:  # Should only be set and used for test & dev
        return (
            CloudConnectorUser(
                authenticated=True, is_debug=True, global_user_id=global_user_id
            ),
            None,
        )

    if AUTH_HEADER.lower() not in headers:
        return CloudConnectorUser(False), CloudConnectorAuthError(
            "No authorization header presented"
        )

    auth_header = headers[AUTH_HEADER.lower()]
    bearer, _, token = auth_header.partition(" ")
    if bearer.lower() != PREFIX_BEARER_HEADER:
        return CloudConnectorUser(False), CloudConnectorAuthError(
            "Invalid authorization header"
        )

    cloud_connector_error, claims = _authenticate_with_token(
        headers, token, oidc_auth_provider
    )
    if cloud_connector_error:
        return (
            CloudConnectorUser(False, claims=claims, global_user_id=global_user_id),
            cloud_connector_error,
        )

    if cloud_connector_error := _validate_headers(claims, headers):
        return (
            CloudConnectorUser(False, claims=claims, global_user_id=global_user_id),
            cloud_connector_error,
        )

    return CloudConnectorUser(True, claims=claims, global_user_id=global_user_id), None


def _authenticate_with_token(
    headers, token, oidc_auth_provider
) -> Tuple[Optional[CloudConnectorAuthError], UserClaims]:
    if headers.get(AUTH_TYPE_HEADER.lower()) != OIDC_AUTH:
        return (
            CloudConnectorAuthError(
                "Invalid authentication token type - only OIDC is supported"
            ),
            UserClaims(),
        )

    user = oidc_auth_provider.authenticate(token)

    if not user.authenticated:
        return CloudConnectorAuthError("Forbidden by auth provider"), UserClaims()

    return None, user.claims


def _validate_headers(claims, headers) -> Optional[CloudConnectorAuthError]:
    claim_header_mapping = {
        "gitlab_realm": X_GITLAB_REALM_HEADER,
        "gitlab_instance_id": X_GITLAB_INSTANCE_ID_HEADER,
        "subject": (
            X_GITLAB_GLOBAL_USER_ID_HEADER
            if claims.issuer == CloudConnectorConfig().service_name
            else X_GITLAB_INSTANCE_ID_HEADER
        ),
    }

    for claim, header in claim_header_mapping.items():
        claim_val = getattr(claims, claim)
        if claim_val and claim_val != headers.get(header.lower()):
            return CloudConnectorAuthError(f"Header mismatch '{header}'")

    duo_seat_count_header = headers.get(X_GITLAB_DUO_SEAT_COUNT_HEADER.lower())
    if error := validate_duo_seat_count_header(claims, duo_seat_count_header):
        # Instead of raising an error, we currently log the error and allow
        # the request to continue. See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/672
        log_exception(Exception(error))
        return None

    return None
