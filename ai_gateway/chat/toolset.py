from ai_gateway.auth import GitLabUser
from ai_gateway.chat.base import BaseToolsRegistry, UnitPrimitiveToolset
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
from ai_gateway.feature_flags import FeatureFlag, is_feature_enabled
from ai_gateway.gitlab_features import GitLabUnitPrimitive, WrongUnitPrimitives

__all__ = ["DuoChatToolsRegistry"]


class DuoChatToolsRegistry(BaseToolsRegistry):
    @property
    def toolsets(self) -> list[UnitPrimitiveToolset]:
        # We can also read the list of tools and associated unit primitives from the file
        # similar to what we implemented for the Prompt Registry
        toolsets = [
            UnitPrimitiveToolset(
                name=GitLabUnitPrimitive.DUO_CHAT,
                min_required_gl_version=None,
                tools=[
                    CiEditorAssistant(),
                ],
            ),
            UnitPrimitiveToolset(
                name=GitLabUnitPrimitive.DOCUMENTATION_SEARCH,
                min_required_gl_version=None,
                tools=[GitlabDocumentation()],
            ),
            UnitPrimitiveToolset(
                name=GitLabUnitPrimitive.ASK_EPIC,
                min_required_gl_version=None,
                tools=[EpicReader()],
            ),
            UnitPrimitiveToolset(
                name=GitLabUnitPrimitive.ASK_ISSUE,
                min_required_gl_version=None,
                tools=[IssueReader()],
            ),
            UnitPrimitiveToolset(
                name=GitLabUnitPrimitive.ASK_MERGE_REQUEST,
                min_required_gl_version="17.5.0-pre",
                tools=[MergeRequestReader()],
            ),
        ]

        if is_feature_enabled(FeatureFlag.AI_COMMIT_READER_FOR_CHAT):
            toolsets.append(
                UnitPrimitiveToolset(
                    name=GitLabUnitPrimitive.ASK_COMMIT,
                    min_required_gl_version="17.5.0-pre",
                    tools=[CommitReader()],
                )
            )

        if is_feature_enabled(FeatureFlag.AI_BUILD_READER_FOR_CHAT):
            toolsets.append(
                UnitPrimitiveToolset(
                    name=GitLabUnitPrimitive.ASK_BUILD,
                    min_required_gl_version="17.5.0-pre",
                    tools=[BuildReader()],
                )
            )

        return toolsets

    def get_on_behalf(
        self, user: GitLabUser, gl_version: str, raise_exception: bool = True
    ) -> list[BaseTool]:
        tools = []
        user_unit_primitives = user.unit_primitives

        for toolset in self.toolsets:
            if toolset.is_available_for(user_unit_primitives, gl_version):
                tools.extend(toolset.tools)

        if len(tools) == 0 and raise_exception:
            raise WrongUnitPrimitives(
                "user doesn't have access to any of the unit primitives"
            )

        return tools

    def get_all(self) -> list[BaseTool]:
        tools = []
        for toolset in self.toolsets:
            tools.extend(toolset.tools)

        return tools
