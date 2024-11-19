import pytest
from gitlab_cloud_connector import (
    CloudConnectorUser,
    GitLabUnitPrimitive,
    UserClaims,
    WrongUnitPrimitives,
)

from ai_gateway.api.auth_utils import StarletteUser
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
            ([GitLabUnitPrimitive.COMPLETE_CODE], ["complete_code"], True),
            (
                [GitLabUnitPrimitive.COMPLETE_CODE, GitLabUnitPrimitive.ASK_BUILD],
                ["complete_code", "ask_build"],
                True,
            ),
            ([GitLabUnitPrimitive.COMPLETE_CODE], [], False),
            (
                [
                    GitLabUnitPrimitive.COMPLETE_CODE,
                    GitLabUnitPrimitive.ASK_BUILD,
                ],
                ["complete_code"],
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
