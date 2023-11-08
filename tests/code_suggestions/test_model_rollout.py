import pytest

from ai_gateway.api.middleware import GitLabUser, UserClaims
from ai_gateway.api.rollout.model import ModelRollout, ModelRolloutWithFallbackPlan


@pytest.mark.parametrize(
    ("rollout_percentage", "project_id", "primary_model", "fallback_model", "model"),
    [
        (
            10,
            None,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GOOGLE_CODE_BISON,
            ModelRollout.GOOGLE_CODE_BISON,
        ),
        (
            0,
            45504304,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GOOGLE_CODE_BISON,
            ModelRollout.GOOGLE_CODE_BISON,
        ),
        (
            5,
            455043,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GOOGLE_CODE_BISON,
            ModelRollout.GOOGLE_CODE_BISON,
        ),
        (
            80,
            45504304,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GOOGLE_CODE_BISON,
            ModelRollout.GOOGLE_CODE_GECKO,
        ),
        (
            80,
            455043,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GOOGLE_CODE_BISON,
            ModelRollout.GOOGLE_CODE_GECKO,
        ),
        (
            80,
            None,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GOOGLE_CODE_BISON,
            ModelRollout.GOOGLE_CODE_BISON,
        ),
        (
            100,
            None,
            ModelRollout.GOOGLE_CODE_GECKO,
            ModelRollout.GOOGLE_CODE_BISON,
            ModelRollout.GOOGLE_CODE_GECKO,
        ),
    ],
)
def test_model_rollout_with_fallback(
    rollout_percentage, project_id, primary_model, fallback_model, model
):
    user = GitLabUser(authenticated=True, claims=UserClaims())

    rollout = ModelRolloutWithFallbackPlan(
        rollout_percentage=rollout_percentage,
        primary_model=primary_model,
        fallback_model=fallback_model,
    )

    assert rollout.route(user, project_id) == model
