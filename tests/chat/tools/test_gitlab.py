from ai_gateway.chat.tools.gitlab import (
    CiEditorAssistant,
    EpicReader,
    GitlabDocumentation,
    GitLabToolkit,
    IssueReader,
)


def test_gitlab_toolkit():
    toolkit = GitLabToolkit()
    tools = [type(tool) for tool in toolkit.get_tools()]

    assert tools == [CiEditorAssistant, IssueReader, GitlabDocumentation, EpicReader]
