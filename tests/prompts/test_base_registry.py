import pytest

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.cloud_connector import CloudConnectorUser, UserClaims
from ai_gateway.gitlab_features import GitLabUnitPrimitive, WrongUnitPrimitives
from ai_gateway.prompts import BasePromptRegistry, Prompt


@pytest.fixture
def user(scopes: list[str]):
    yield StarletteUser(
        CloudConnectorUser(authenticated=True, claims=UserClaims(scopes=scopes))
    )


@pytest.fixture
def registry(prompt: Prompt):
    class Registry(BasePromptRegistry):
        def get(self, *args, **kwargs):
            return prompt

    yield Registry()


class TestBaseRegistry:
    @pytest.mark.parametrize(
        ("unit_primitives", "scopes", "success"),
        [
            ([GitLabUnitPrimitive.CODE_SUGGESTIONS], ["code_suggestions"], True),
            (
                [GitLabUnitPrimitive.CODE_SUGGESTIONS, GitLabUnitPrimitive.ASK_BUILD],
                ["code_suggestions", "ask_build"],
                True,
            ),
            ([GitLabUnitPrimitive.CODE_SUGGESTIONS], [], False),
            (
                [
                    GitLabUnitPrimitive.CODE_SUGGESTIONS,
                    GitLabUnitPrimitive.ASK_BUILD,
                ],
                ["code_suggestions"],
                False,
            ),
        ],
    )
    def test_get_on_behalf(
        self,
        registry: BasePromptRegistry,
        user: StarletteUser,
        prompt: Prompt,
        unit_primitives: list[GitLabUnitPrimitive],
        scopes: list[str],
        success: bool,
    ):
        if success:
            assert registry.get_on_behalf(user=user, prompt_id="test") == prompt
        else:
            with pytest.raises(WrongUnitPrimitives):
                registry.get_on_behalf(user=user, prompt_id="test")
