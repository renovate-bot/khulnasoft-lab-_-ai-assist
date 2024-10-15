import pytest

from ai_gateway.cloud_connector import (
    AuthProvider,
    CloudConnectorUser,
    User,
    UserClaims,
)
from ai_gateway.cloud_connector import authenticate as cloud_connector_authenticate


class StubAuthProviderBypassAuth(AuthProvider):
    def authenticate(self, token):
        return None


@pytest.mark.parametrize(
    (
        "headers",
        "expected_gitlab_global_user_id",
    ),
    [({}, None), ({"X-Gitlab-Global-User-Id": "12345"}, "12345")],
)
def test_cloud_connector_authenticate_bypass_auth(
    headers, expected_gitlab_global_user_id
):
    cloud_connector_user, cloud_connector_error = cloud_connector_authenticate(
        headers, StubAuthProviderBypassAuth(), bypass_auth=True
    )
    assert vars(cloud_connector_user) == vars(
        CloudConnectorUser(
            authenticated=True,
            is_debug=True,
            global_user_id=expected_gitlab_global_user_id,
            claims=None,
        )
    )
    assert cloud_connector_error is None


class StubAuthProvider(AuthProvider):
    def __init__(self, user):
        self.user = user

    def authenticate(self, token):
        return self.user


@pytest.mark.parametrize(
    (
        "headers",
        "auth_provider_authenticated",
        "expected_user",
        "expected_error",
    ),
    [
        # Missing Authorization header
        (
            {},
            False,
            CloudConnectorUser(
                authenticated=False, is_debug=False, global_user_id=None, claims=None
            ),
            "No authorization header presented",
        ),
        # Missing Authorization header, X-Gitlab-Global-User-Id header is not read
        (
            {"X-Gitlab-Global-User-Id": "1111"},
            False,
            CloudConnectorUser(
                authenticated=False, is_debug=False, global_user_id=None, claims=None
            ),
            "No authorization header presented",
        ),
        # Invalid Authorization header value
        (
            {"Authorization": "invalid", "X-Gitlab-Global-User-Id": "1111"},
            False,
            CloudConnectorUser(
                authenticated=False, is_debug=False, global_user_id=None, claims=None
            ),
            "Invalid authorization header",
        ),
        # Invalid X-Gitlab-Authentication-Type header value
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "INVALID",
                "X-Gitlab-Global-User-Id": "1111",
            },
            False,
            CloudConnectorUser(
                authenticated=False,
                is_debug=False,
                global_user_id="1111",
                claims=UserClaims(),
            ),
            "Invalid authentication token type - only OIDC is supported",
        ),
        # Valid headers format. AuthProvider returned authenticated=False (invalid token)
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1111",
            },
            False,
            CloudConnectorUser(
                authenticated=False,
                is_debug=False,
                global_user_id="1111",
                claims=UserClaims(),
            ),
            "Forbidden by auth provider",
        ),
        # Missing X-Gitlab-Duo-Seat-Count, but we have a duo_seat_count claim
        # It's an exceptional case (temporarily while investigating claim <-> header discrepancy)
        # We log an error, but we still want the request to pass.
        # See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/672
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1111",
                "X-Gitlab-Instance-Id": "3333",
                "X-GitLab-Realm": "self-managed",
            },
            True,
            CloudConnectorUser(
                authenticated=True,
                is_debug=False,
                global_user_id="1111",
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="3333",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="3333",
                    duo_seat_count="777",
                ),
            ),
            None,  # "Header is missing: 'X-Gitlab-Duo-Seat-Count'",
        ),
        # Auth OK: Missing X-Gitlab-Duo-Seat-Count, and we DO NOT have a duo_seat_count claim
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1111",
                "X-Gitlab-Instance-Id": "3333",
                "X-GitLab-Realm": "self-managed",
            },
            True,
            CloudConnectorUser(
                authenticated=True,
                is_debug=False,
                global_user_id="1111",
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="3333",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="3333",
                ),
            ),
            None,
        ),
        # Header mismatch: X-Gitlab-Duo-Seat-Count
        # It's an exceptional case (temporarily while investigating claim <-> header discrepancy)
        # We log an error, but we still want the request to pass.
        # See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/672
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1111",
                "X-Gitlab-Instance-Id": "3333",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Duo-Seat-Count": "1000",
            },
            True,
            CloudConnectorUser(
                authenticated=True,
                is_debug=False,
                global_user_id="1111",
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="3333",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="3333",
                    duo_seat_count="777",
                ),
            ),
            None,  # "Header mismatch 'X-Gitlab-Duo-Seat-Count'",
        ),
        # Header mismatch: X-Gitlab-Instance-Id
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1111",
                "X-Gitlab-Instance-Id": "mismatch",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Duo-Seat-Count": "777",
            },
            True,
            CloudConnectorUser(
                authenticated=False,
                is_debug=False,
                global_user_id="1111",
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="3333",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="3333",
                    duo_seat_count="777",
                ),
            ),
            "Header mismatch 'X-Gitlab-Instance-Id'",
        ),
        # Header mismatch: X-Gitlab-Realm
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1111",
                "X-Gitlab-Instance-Id": "3333",
                "X-GitLab-Realm": "mismatch",
                "X-Gitlab-Duo-Seat-Count": "777",
            },
            True,
            CloudConnectorUser(
                authenticated=False,
                is_debug=False,
                global_user_id="1111",
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="3333",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="3333",
                    duo_seat_count="777",
                ),
            ),
            "Header mismatch 'X-Gitlab-Realm'",
        ),
        # Header mismatch: X-Gitlab-Global-User-Id
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "mismatch",
                "X-Gitlab-Instance-Id": "3333",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Duo-Seat-Count": "777",
            },
            True,
            CloudConnectorUser(
                authenticated=False,
                is_debug=False,
                global_user_id="mismatch",
                claims=UserClaims(
                    issuer="gitlab-ai-gateway",
                    scopes=["feature1", "feature3"],
                    subject="1111",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="3333",
                    duo_seat_count="777",
                ),
            ),
            "Header mismatch 'X-Gitlab-Global-User-Id'",
        ),
        # Auth OK
        (
            {
                "Authorization": "Bearer 2222",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1111",
                "X-Gitlab-Instance-Id": "3333",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Duo-Seat-Count": "777",
            },
            True,
            CloudConnectorUser(
                authenticated=True,
                is_debug=False,
                global_user_id="1111",
                claims=UserClaims(
                    scopes=["feature1", "feature3"],
                    subject="3333",
                    gitlab_realm="self-managed",
                    gitlab_instance_id="3333",
                    duo_seat_count="777",
                ),
            ),
            None,
        ),
    ],
)
def test_cloud_connector_authenticate(
    headers, auth_provider_authenticated, expected_user, expected_error
):
    auth_provider_result = User(
        authenticated=auth_provider_authenticated, claims=expected_user.claims
    )
    cloud_connector_user, cloud_connector_error = cloud_connector_authenticate(
        headers, StubAuthProvider(auth_provider_result)
    )
    assert vars(cloud_connector_user) == vars(expected_user)
    assert getattr(cloud_connector_error, "error_message", None) == expected_error
