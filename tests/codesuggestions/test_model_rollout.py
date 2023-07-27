import pytest

from codesuggestions.api.middleware import GitLabUser, UserClaims
from codesuggestions.api.rollout.model import (
    ModelRollout,
    ModelRolloutThirdPartyPlan,
    ModelRolloutWithFallbackPlan,
)
from codesuggestions.config import Project

THIRD_PARTY_AI_MODELS = [
    ModelRollout.GOOGLE_TEXT_BISON,
    ModelRollout.GOOGLE_CODE_BISON,
    ModelRollout.GOOGLE_CODE_GECKO,
]


@pytest.mark.parametrize(
    (
        "rollout_percentage",
        "project_id",
        "model",
        "f_third_party_ai_default",
        "is_third_party_ai_default",
    ),
    [
        (10, None, ModelRollout.GITLAB_CODEGEN, True, True),
        (0, 45504304, ModelRollout.GITLAB_CODEGEN, True, True),
        (80, 45504304, ModelRollout.GOOGLE_TEXT_BISON, True, True),
        (80, 40916776, ModelRollout.GOOGLE_CODE_BISON, True, True),
        (
            80,
            2670515,
            ModelRollout.GITLAB_CODEGEN,
            True,
            True,
        ),  # Does not fall in the rollout window
        (80, 1507906, ModelRollout.GOOGLE_CODE_GECKO, True, True),
        (
            100,
            89283912,
            ModelRollout.GOOGLE_CODE_GECKO,
            True,
            True,
        ),  # Does not change when rollout percentage increase
        (
            80,
            45504304,
            ModelRollout.GITLAB_CODEGEN,
            False,
            False,
        ),  # third-party models disabled
        (
            80,
            45504302,
            ModelRollout.GOOGLE_CODE_GECKO,
            False,
            False,
        ),  # Should be codegen, but user has limited 3rd party access
    ],
)
def test_model_router(
    rollout_percentage,
    project_id,
    model,
    f_third_party_ai_default,
    is_third_party_ai_default,
):
    user = GitLabUser(
        authenticated=True,
        claims=UserClaims(is_third_party_ai_default=is_third_party_ai_default),
    )

    router = ModelRolloutThirdPartyPlan(
        rollout_percentage=rollout_percentage,
        f_third_party_ai_default=f_third_party_ai_default,
        f_limited_access_third_party_ai={45504302: Project(45504302, "")},
        third_party_ai_models=THIRD_PARTY_AI_MODELS,
    )

    assert router.route(user, project_id) == model


@pytest.mark.parametrize(
    ("project_id", "limited_access_ids", "want"),
    [
        (123, {}, False),  # No IDs in the `f_limited_access_third_party_ai`
        (
            123,
            {111: Project(111, "")},
            False,
        ),  # Project ID is not in `f_limited_access_third_party_ai`
        (123, {123: Project(123, "")}, True),
    ],
)
def test_is_third_party_ai_limited_access(project_id, limited_access_ids, want):
    router = ModelRolloutThirdPartyPlan(
        rollout_percentage=50,
        f_third_party_ai_default=True,
        f_limited_access_third_party_ai=limited_access_ids,
        third_party_ai_models=THIRD_PARTY_AI_MODELS,
    )

    assert router._is_third_party_ai_limited_access(project_id) == want


@pytest.mark.parametrize(
    ("user", "f_third_party_ai_default", "want"),
    [
        (
            GitLabUser(
                authenticated=True,
                is_debug=True,
                claims=UserClaims(is_third_party_ai_default=False),
            ),
            True,
            True,
        ),
        (
            GitLabUser(
                authenticated=True,
                is_debug=False,
                claims=UserClaims(is_third_party_ai_default=True),
            ),
            True,
            True,
        ),
        (
            GitLabUser(
                authenticated=True,
                is_debug=True,
                claims=UserClaims(is_third_party_ai_default=True),
            ),
            True,
            True,
        ),
        (
            GitLabUser(
                authenticated=True,
                is_debug=True,
                claims=UserClaims(is_third_party_ai_default=True),
            ),
            False,
            False,
        ),
    ],
)
def test_resolve_third_party_ai_flag(user, f_third_party_ai_default, want):
    router = ModelRolloutThirdPartyPlan(
        rollout_percentage=50,
        f_third_party_ai_default=f_third_party_ai_default,
        f_limited_access_third_party_ai={},
        third_party_ai_models=THIRD_PARTY_AI_MODELS,
    )

    assert router._resolve_third_party_ai_flag(user) == want


@pytest.mark.parametrize(
    ("rollout_percentage", "project_id", "primary_model", "fallback_model", "model"),
    [
        (
            10,
            None,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GITLAB_CODEGEN,
            ModelRollout.GITLAB_CODEGEN,
        ),
        (
            0,
            45504304,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GITLAB_CODEGEN,
            ModelRollout.GITLAB_CODEGEN,
        ),
        (
            5,
            455043,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GITLAB_CODEGEN,
            ModelRollout.GITLAB_CODEGEN,
        ),
        (
            80,
            45504304,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GITLAB_CODEGEN,
            ModelRollout.GOOGLE_CODE_GECKO,
        ),
        (
            80,
            455043,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GITLAB_CODEGEN,
            ModelRollout.GOOGLE_CODE_GECKO,
        ),
        (
            80,
            None,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GITLAB_CODEGEN,
            ModelRollout.GITLAB_CODEGEN,
        ),
        (
            100,
            None,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GITLAB_CODEGEN,
            ModelRollout.GOOGLE_CODE_GECKO,
        ),
    ],
)
def test_model_rollout_with_fallback(
    rollout_percentage, project_id, primary_model, fallback_model, model
):
    user = GitLabUser(
        authenticated=True, claims=UserClaims(is_third_party_ai_default=False)
    )

    rollout = ModelRolloutWithFallbackPlan(
        rollout_percentage=rollout_percentage,
        primary_model=primary_model,
        fallback_model=fallback_model,
    )

    assert rollout.route(user, project_id) == model
