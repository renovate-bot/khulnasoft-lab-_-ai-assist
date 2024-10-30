# flake8: noqa

from ai_gateway.cloud_connector.auth import CloudConnectorAuthError, authenticate
from ai_gateway.cloud_connector.cache import LocalAuthCache
from ai_gateway.cloud_connector.config import CloudConnectorConfig
from ai_gateway.cloud_connector.gitlab_features import (
    FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    UNIT_PRIMITIVE_AND_DESCRIPTION_MAPPING,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
    WrongUnitPrimitives,
)
from ai_gateway.cloud_connector.logging import log_exception
from ai_gateway.cloud_connector.providers import (
    AuthProvider,
    CompositeProvider,
    GitLabOidcProvider,
    LocalAuthProvider,
)
from ai_gateway.cloud_connector.token_authority import TokenAuthority
from ai_gateway.cloud_connector.user import CloudConnectorUser, UserClaims
from ai_gateway.cloud_connector.validators import (
    X_GITLAB_DUO_SEAT_COUNT_HEADER,
    validate_duo_seat_count_header,
)
