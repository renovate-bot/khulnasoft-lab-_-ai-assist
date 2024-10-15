from typing import Type

import pytest

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.chat.tools import BaseTool
from ai_gateway.chat.tools.gitlab import (
    BuildReader,
    CiEditorAssistant,
    CommitReader,
    EpicReader,
    GitlabDocumentation,
    IssueReader,
    MergeRequestReader,
)
from ai_gateway.chat.toolset import DuoChatToolsRegistry
from ai_gateway.cloud_connector import CloudConnectorUser, UserClaims
from ai_gateway.feature_flags.context import current_feature_flag_context
from ai_gateway.gitlab_features import GitLabUnitPrimitive, WrongUnitPrimitives


class TestDuoChatToolRegistry:
    @pytest.mark.parametrize(
        "expected_tools",
        [
            (
                [
                    CiEditorAssistant,
                    GitlabDocumentation,
                    EpicReader,
                    IssueReader,
                    MergeRequestReader,
                ]
            )
        ],
    )
    def test_get_all_success(self, expected_tools: list[Type[BaseTool]]):
        tools = DuoChatToolsRegistry().get_all()
        actual_tools = [type(tool) for tool in tools]

        assert actual_tools == expected_tools

    @pytest.mark.parametrize(
        ("unit_primitives", "expected_tools"),
        [
            ([GitLabUnitPrimitive.DUO_CHAT], [CiEditorAssistant]),
            ([GitLabUnitPrimitive.DOCUMENTATION_SEARCH], [GitlabDocumentation]),
            ([GitLabUnitPrimitive.ASK_EPIC], [EpicReader]),
            ([GitLabUnitPrimitive.ASK_ISSUE], [IssueReader]),
            (
                [
                    GitLabUnitPrimitive.DUO_CHAT,
                    GitLabUnitPrimitive.DOCUMENTATION_SEARCH,
                    GitLabUnitPrimitive.ASK_EPIC,
                    GitLabUnitPrimitive.ASK_ISSUE,
                ],
                [
                    CiEditorAssistant,
                    GitlabDocumentation,
                    EpicReader,
                    IssueReader,
                ],
            ),
            (
                [GitLabUnitPrimitive.CODE_SUGGESTIONS],
                [],
            ),
        ],
    )
    def test_get_on_behalf_success(
        self,
        unit_primitives: list[GitLabUnitPrimitive],
        expected_tools: list[Type[BaseTool]],
    ):
        user = StarletteUser(
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(scopes=[u.value for u in unit_primitives]),
            )
        )

        tools = DuoChatToolsRegistry().get_on_behalf(user, "", raise_exception=False)
        actual_tools = [type(tool) for tool in tools]

        assert actual_tools == expected_tools

    @pytest.mark.parametrize(
        "unit_primitives",
        [([GitLabUnitPrimitive.CODE_SUGGESTIONS, GitLabUnitPrimitive.EXPLAIN_CODE])],
    )
    def test_get_on_behalf_error(
        self,
        unit_primitives: list[GitLabUnitPrimitive],
    ):
        user = StarletteUser(
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(scopes=[u.value for u in unit_primitives]),
            )
        )

        with pytest.raises(WrongUnitPrimitives):
            DuoChatToolsRegistry().get_on_behalf(user, "", raise_exception=True)

    @pytest.mark.parametrize(
        "feature_flag, unit_primitive, reader_tool_type",
        [
            ("ai_commit_reader_for_chat", GitLabUnitPrimitive.ASK_COMMIT, CommitReader),
            ("ai_build_reader_for_chat", GitLabUnitPrimitive.ASK_BUILD, BuildReader),
        ],
    )
    def test_feature_flag(
        self,
        feature_flag: str,
        unit_primitive: GitLabUnitPrimitive,
        reader_tool_type: Type[BaseTool],
    ):
        current_feature_flag_context.set({feature_flag})

        user = StarletteUser(
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(
                    scopes=[
                        unit_primitive.value,
                        GitLabUnitPrimitive.DUO_CHAT.value,
                    ]
                ),
            )
        )

        tools = DuoChatToolsRegistry().get_on_behalf(
            user, "17.5.0-pre", raise_exception=False
        )
        actual_tools = [type(tool) for tool in tools]

        assert actual_tools == [CiEditorAssistant, reader_tool_type]

        current_feature_flag_context.set(set())

        tools = DuoChatToolsRegistry().get_on_behalf(
            user, "17.5.0-pre", raise_exception=False
        )
        actual_tools = [type(tool) for tool in tools]

        assert actual_tools == [CiEditorAssistant]
