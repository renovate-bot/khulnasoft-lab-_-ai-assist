from textwrap import dedent

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

    description: str = dedent(
        """\
        Useful tool when you need to provide suggestions regarding anything related
        to ".gitlab-ci.yml" file. It helps with questions related to deploying code, configuring CI/CD pipelines,
        defining CI jobs, or environments.
        It can not help with writing code in general or questions about software development."""
    )

    example: str = dedent(
        """\
        Question: Please create a deployment configuration for a node.js application.
        Thought: You have asked a question related to deployment of an application or CI/CD pipelines.
            "ci_editor_assistant" tool can assist with this kind of questions.
        Action: ci_editor_assistant
        Action Input: Please create a deployment configuration for a node.js application."""
    )


class IssueReader(BaseRemoteTool):
    name: str = "issue_reader"
    resource: str = "issues"

    description: str = dedent(
        """\
        This tool retrieves the content of a specific issue
        ONLY if the user question fulfills the strict usage conditions below.

        **Strict Usage Conditions:**
        * **Condition 1: Issue ID Provided:** This tool MUST be used ONLY when the user provides a valid issue ID.
        * **Condition 2: Issue URL Context:** This tool MUST be used ONLY when the user is actively viewing a specific
          issue URL or a specific URL is provided by the user.

        **Do NOT** attempt to search for or identify issues based on descriptions, keywords, or user questions.

        **Action Input:**
        * The original question asked by the user.

        **Important:**  Reject any input that does not strictly adhere to the usage conditions above.
        Return a message stating you are unable to search for issues without a valid identifier."""
    )

    example: str = dedent(
        """\
        Question: Please identify the author of #123 issue
        Thought: You have access to the same resources as user who asks a question.
          Question is about the content of an issue, so you need to use "issue_reader" tool to retrieve and read issue.
          Based on this information you can present final answer about issue.
        Action: issue_reader
        Action Input: Please identify the author of #123 issue"""
    )


class GitlabDocumentation(BaseRemoteTool):
    name: str = "gitlab_documentation"
    resource: str = "documentation answers"

    description: str = dedent(
        """\
        This tool is beneficial when you need to answer questions concerning GitLab and its features.
        Questions can be about GitLab's projects, groups, issues, merge requests,
        epics, milestones, labels, CI/CD pipelines, git repositories, and more."""
    )

    example: str = dedent(
        """\
        Question: How do I set up a new project?
        Thought: Question is about inner working of GitLab. "gitlab_documentation" tool is the right one for the job.
        Action: gitlab_documentation
        Action Input: How do I set up a new project?"""
    )


class EpicReader(BaseRemoteTool):
    name: str = "epic_reader"
    resource: str = "epics"

    description: str = dedent(
        """\
        This tool retrieves the content of a specific epic
        ONLY if the user question fulfills the strict usage conditions below.

        **Strict Usage Conditions:**
        * **Condition 1: epic ID Provided:** This tool MUST be used ONLY when the user provides a valid epic ID.
        * **Condition 2: epic URL Context:** This tool MUST be used ONLY when the user is actively viewing
          a specific epic URL or a specific URL is provided by the user.

        **Do NOT** attempt to search for or identify epics based on descriptions, keywords, or user questions.

        **Action Input:**
        * The original question asked by the user.

        **Important:**  Reject any input that does not strictly adhere to the usage conditions above.
        Return a message stating you are unable to search for epics without a valid identifier."""
    )

    example: str = dedent(
        """\
        Question: Please identify the author of &123 epic.
        Thought: You have access to the same resources as user who asks a question.
            The question is about an epic, so you need to use "epic_reader" tool.
            Based on this information you can present final answer.
        Action: epic_reader
        Action Input: Please identify the author of &123 epic."""
    )
