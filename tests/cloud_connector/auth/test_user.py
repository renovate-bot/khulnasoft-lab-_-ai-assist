import pytest

from ai_gateway.cloud_connector import (
    CloudConnectorUser,
    GitLabUnitPrimitive,
    UserClaims,
)


@pytest.fixture
def user():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["code_suggestions"],
            subject="1234",
            gitlab_realm="self-managed",
            issuer="https://customers.gitlab.com",
        ),
        global_user_id="test-user-id",
    )


def test_issuer_not_in_disallowed_issuers(user: CloudConnectorUser):
    assert user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS, disallowed_issuers=["gitlab-ai-gateway"]
    )


def test_issuer_in_disallowed_issuers(user: CloudConnectorUser):
    assert not user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=["https://customers.gitlab.com"],
    )


@pytest.mark.parametrize(
    ("user", "expected_unit_primitives"),
    [
        (
            CloudConnectorUser(
                authenticated=True, claims=UserClaims(scopes=["code_suggestions"])
            ),
            [GitLabUnitPrimitive.CODE_SUGGESTIONS],
        ),
        (
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(scopes=["code_suggestions", "unknown"]),
            ),
            [GitLabUnitPrimitive.CODE_SUGGESTIONS],
        ),
        (CloudConnectorUser(authenticated=True, claims=None), []),
    ],
)
def test_user_unit_primitives(
    user: CloudConnectorUser,
    expected_unit_primitives: list[GitLabUnitPrimitive],
):
    actual_unit_primitives = user.unit_primitives

    assert actual_unit_primitives == expected_unit_primitives
