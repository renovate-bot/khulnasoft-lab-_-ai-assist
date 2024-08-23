import pytest

from ai_gateway.chat.base import UnitPrimitiveToolset
from ai_gateway.gitlab_features import GitLabUnitPrimitive


@pytest.mark.parametrize(
    ("unit_primitives", "min_required_gl_version", "gl_version", "is_available"),
    [
        ([GitLabUnitPrimitive.DOCUMENTATION_SEARCH], None, "", False),
        (
            [GitLabUnitPrimitive.DUO_CHAT, GitLabUnitPrimitive.DOCUMENTATION_SEARCH],
            None,
            "",
            True,
        ),
        ([GitLabUnitPrimitive.DUO_CHAT], "17.2.0", "", False),
        ([GitLabUnitPrimitive.DUO_CHAT], "17.2.0", "invalid-version", False),
        ([GitLabUnitPrimitive.DUO_CHAT], "17.2.0", "17.1.0", False),
        ([GitLabUnitPrimitive.DUO_CHAT], "17.2.0", "17.3.0", True),
    ],
)
def test_is_available_for(
    unit_primitives: list[GitLabUnitPrimitive],
    min_required_gl_version: str,
    gl_version: str,
    is_available: bool,
):
    toolset = UnitPrimitiveToolset(
        name=GitLabUnitPrimitive.DUO_CHAT,
        min_required_gl_version=min_required_gl_version,
        tools=[],
    )

    assert toolset.is_available_for(unit_primitives, gl_version) == is_available
