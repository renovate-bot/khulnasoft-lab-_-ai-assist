# flake8: noqa

from ai_gateway.cloud_connector.auth import CloudConnectorAuthError, authenticate
from ai_gateway.cloud_connector.cache import LocalAuthCache
from ai_gateway.cloud_connector.logging import log_exception
from ai_gateway.cloud_connector.providers import (
    AuthProvider,
    CompositeProvider,
    GitLabOidcProvider,
    LocalAuthProvider,
)
from ai_gateway.cloud_connector.token_authority import (
    SELF_SIGNED_TOKEN_ISSUER,
    TokenAuthority,
)
from ai_gateway.cloud_connector.user import CloudConnectorUser, User, UserClaims
from ai_gateway.cloud_connector.validators import (
    X_GITLAB_DUO_SEAT_COUNT_HEADER,
    validate_duo_seat_count_header,
)
