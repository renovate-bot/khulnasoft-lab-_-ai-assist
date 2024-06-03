import pytest

from ai_gateway.auth.user import GitLabUser, UserClaims
from ai_gateway.gitlab_features import GitLabUnitPrimitive


@pytest.fixture
def user():
    return GitLabUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["code_suggestions"],
            subject="1234",
            gitlab_realm="self-managed",
            issuer="https://customers.gitlab.com",
        ),
    )


def test_issuer_not_in_disallowed_issuers(user: GitLabUser):
    assert user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS, disallowed_issuers=["gitlab-ai-gateway"]
    )


def test_issuer_in_disallowed_issuers(user: GitLabUser):
    assert not user.can(
        GitLabUnitPrimitive.CODE_SUGGESTIONS,
        disallowed_issuers=["https://customers.gitlab.com"],
    )
