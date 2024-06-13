from ai_gateway.chat.tools.base import BaseRemoteTool

__all__ = [
    "CiEditorAssistant",
    "IssueReader",
    "GitlabDocumentation",
    "EpicReader",
]


class CiEditorAssistant(BaseRemoteTool):
    name: str = "ci_editor_assistant"
    resource: str = "ci editor answers"

    description: str = """Useful tool when you need to provide suggestions regarding anything related
        to ".gitlab-ci.yml" file. It helps with questions related to deploying code, configuring CI/CD pipelines,
        defining CI jobs, or environments."""

    example: str = """Question: Please create a deployment configuration for a node.js application.
        Thought: You have asked a question related to deployment of an application or CI/CD pipelines.
            "ci_editor_assistant" tool can assist with this kind of questions.
        Action: ci_editor_assistant
        Action Input: Please create a deployment configuration for a node.js application."""


class IssueReader(BaseRemoteTool):
    name: str = "issue_reader"
    resource: str = "issues"

    description: str = """Gets the content of the current issue (also referenced as this or that) the user sees
        or a specific issue identified by an ID or a URL. In this context, word `issue` means core building block
        in GitLab that enable collaboration, discussions, planning and tracking of work.
        Action Input for this tool should be the original question or issue identifier."""

    example: str = """Question: Please identify the author of #123 issue
        Thought: You have access to the same resources as user who asks a question.
            Question is about the content of an issue, so you need to use "issue_reader" tool to retrieve and read issue.
            Based on this information you can present final answer about issue.
        Action: issue_reader"
        Action Input: Please identify the author of #123 issue"""


class GitlabDocumentation(BaseRemoteTool):
    name: str = "gitlab_documentation"
    resource: str = "documentation answers"

    description: str = """This tool is beneficial when you need to answer questions concerning GitLab and its features.
        Questions can be about GitLab's projects, groups, issues, merge requests,
        epics, milestones, labels, CI/CD pipelines, git repositories, and more."""

    example: str = """Question: How do I set up a new project?
        Thought: Question is about inner working of GitLab. "gitlab_documentation" tool is the right one for the job.
        Action: gitlab_documentation
        Action Input: How do I set up a new project?"""


class EpicReader(BaseRemoteTool):
    name: str = "epic_reader"
    resource: str = "epics"

    description: str = """Useful tool when you need to retrieve information about a specific epic.
        In this context, word `epic` means high-level building block in GitLab that encapsulates high-level plans
        and discussions. Epic can contain multiple issues. Action Input for this tool should be the original
        question or epic identifier."""

    example: str = """Question: Please identify the author of &123 epic.
        Thought: You have access to the same resources as user who asks a question.
            The question is about an epic, so you need to use "epic_reader" tool.
            Based on this information you can present final answer.
        Action: epic_reader
        Action Input: Please identify the author of &123 epic."""
