from enum import StrEnum


# TODO: Validate that these categories exist in https://gitlab.com/gitlab-com/www-gitlab-com/raw/master/data/stages.yml.
class GitLabFeatureCategory(StrEnum):
    AI_ABSTRACTION_LAYER = "ai_abstraction_layer"
    CODE_REVIEW_WORKFLOW = "code_review_workflow"
    CODE_SUGGESTIONS = "code_suggestions"
    CONTINUOUS_INTEGRATION = "continuous_integration"
    DUO_CHAT = "duo_chat"
    GLOBAL_SEARCH = "global_search"
    PRODUCT_ANALYTICS_VISUALIZATION = "product_analytics_visualization"
    SOURCE_CODE_MANAGEMENT = "source_code_management"
    TEAM_PLANNING = "team_planning"
    VULNERABILITY_MANAGEMENT = "vulnerability_management"


# Make sure these unit primitives are defined in `ee/config/cloud_connector/access_data.yml`
class GitLabUnitPrimitive(StrEnum):
    ASK_BUILD = "ask_build"
    ASK_COMMIT = "ask_commit"
    ASK_EPIC = "ask_epic"
    ASK_ISSUE = "ask_issue"
    ASK_MERGE_REQUEST = "ask_merge_request"
    CATEGORIZE_DUO_CHAT_QUESTION = "categorize_duo_chat_question"
    CODE_SUGGESTIONS = "code_suggestions"
    COMPLETE_CODE = "complete_code"
    DOCUMENTATION_SEARCH = "documentation_search"
    DUO_CHAT = "duo_chat"
    DUO_WORKFLOW_GENERATE_TOKEN = "duo_workflow_generate_token"
    DUO_WORKFLOW_EXECUTE_WORKFLOW = "duo_workflow_execute_workflow"
    EXPLAIN_CODE = "explain_code"
    EXPLAIN_VULNERABILITY = "explain_vulnerability"
    FILL_IN_MERGE_REQUEST_TEMPLATE = "fill_in_merge_request_template"
    FIX_CODE = "fix_code"
    GENERATE_CODE = "generate_code"
    GENERATE_COMMIT_MESSAGE = "generate_commit_message"
    GENERATE_CUBE_QUERY = "generate_cube_query"
    GENERATE_ISSUE_DESCRIPTION = "generate_issue_description"
    GLAB_ASK_GIT_COMMAND = "glab_ask_git_command"
    INCLUDE_FILE_CONTEXT = "include_file_context"
    INCLUDE_ISSUE_CONTEXT = "include_issue_context"
    INCLUDE_MERGE_REQUEST_CONTEXT = "include_merge_request_context"
    INCLUDE_SNIPPET_CONTEXT = "include_snippet_context"
    INCLUDE_DEPENDENCY_CONTEXT = "include_dependency_context"
    REFACTOR_CODE = "refactor_code"
    RESOLVE_VULNERABILITY = "resolve_vulnerability"
    REVIEW_MERGE_REQUEST = "review_merge_request"
    SEMANTIC_SEARCH_ISSUE = "semantic_search_issue"
    SUMMARIZE_ISSUE_DISCUSSIONS = "summarize_issue_discussions"
    SUMMARIZE_MERGE_REQUEST = "summarize_merge_request"
    SUMMARIZE_REVIEW = "summarize_review"
    SUMMARIZE_SUBMITTED_REVIEW = "summarize_submitted_review"
    SUMMARIZE_COMMENTS = "summarize_comments"
    TROUBLESHOOT_JOB = "troubleshoot_job"
    WRITE_TESTS = "write_tests"


class WrongUnitPrimitives(Exception):
    pass


FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS = {
    GitLabUnitPrimitive.ASK_BUILD: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.ASK_COMMIT: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.ASK_EPIC: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.ASK_ISSUE: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.ASK_MERGE_REQUEST: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.CATEGORIZE_DUO_CHAT_QUESTION: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.DOCUMENTATION_SEARCH: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.DUO_CHAT: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.EXPLAIN_CODE: GitLabFeatureCategory.SOURCE_CODE_MANAGEMENT,
    GitLabUnitPrimitive.EXPLAIN_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
    GitLabUnitPrimitive.FILL_IN_MERGE_REQUEST_TEMPLATE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.FIX_CODE: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.GENERATE_COMMIT_MESSAGE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.GENERATE_CUBE_QUERY: GitLabFeatureCategory.PRODUCT_ANALYTICS_VISUALIZATION,
    GitLabUnitPrimitive.GENERATE_ISSUE_DESCRIPTION: GitLabFeatureCategory.TEAM_PLANNING,
    GitLabUnitPrimitive.GLAB_ASK_GIT_COMMAND: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.REFACTOR_CODE: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.RESOLVE_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
    GitLabUnitPrimitive.REVIEW_MERGE_REQUEST: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SEMANTIC_SEARCH_ISSUE: GitLabFeatureCategory.GLOBAL_SEARCH,
    GitLabUnitPrimitive.SUMMARIZE_ISSUE_DISCUSSIONS: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SUMMARIZE_MERGE_REQUEST: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SUMMARIZE_REVIEW: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SUMMARIZE_SUBMITTED_REVIEW: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.TROUBLESHOOT_JOB: GitLabFeatureCategory.CONTINUOUS_INTEGRATION,
    GitLabUnitPrimitive.SUMMARIZE_COMMENTS: GitLabFeatureCategory.TEAM_PLANNING,
    GitLabUnitPrimitive.WRITE_TESTS: GitLabFeatureCategory.DUO_CHAT,
}

# TODO: Move these metadata to cloud connector yamls, which are accessible via the cloud connector python client.
# See https://gitlab.com/gitlab-org/gitlab/-/issues/465221
# TODO: Ask stage groups to give better descriptions for these UPs.
UNIT_PRIMITIVE_AND_DESCRIPTION_MAPPING = {
    GitLabUnitPrimitive.ASK_BUILD: "Ask questions about GitLab builds",
    GitLabUnitPrimitive.ASK_COMMIT: "Ask questions about GitLab commits",
    GitLabUnitPrimitive.ASK_EPIC: "Ask questions about GitLab epics",
    GitLabUnitPrimitive.ASK_ISSUE: "Ask questions about GitLab issues",
    GitLabUnitPrimitive.ASK_MERGE_REQUEST: "Ask questions about GitLab merge requests",
    GitLabUnitPrimitive.CATEGORIZE_DUO_CHAT_QUESTION: "Categorize user's question to GitLab Duo Chat for internal telemetry purpose",
    GitLabUnitPrimitive.DOCUMENTATION_SEARCH: "Perform semantic search on gitlab documentations for a given query.",
    GitLabUnitPrimitive.DUO_CHAT: "Ask various GitLab-related questions and tasks.",
    GitLabUnitPrimitive.EXPLAIN_CODE: "Explain function or method of the selected code",
    GitLabUnitPrimitive.EXPLAIN_VULNERABILITY: "Explain a security vulnerability of the given CVE or code.",
    GitLabUnitPrimitive.FILL_IN_MERGE_REQUEST_TEMPLATE: "Fill code change summary in a description of new merge request.",
    GitLabUnitPrimitive.FIX_CODE: "Fix the selected code.",
    GitLabUnitPrimitive.GENERATE_COMMIT_MESSAGE: "Generate a Git commit message.",
    GitLabUnitPrimitive.GENERATE_CUBE_QUERY: "Convert plain text questions about event data in to a structured query in JSON format.",
    GitLabUnitPrimitive.GENERATE_ISSUE_DESCRIPTION: "Generate an issue description.",
    GitLabUnitPrimitive.REFACTOR_CODE: "Refactor the selected code.",
    GitLabUnitPrimitive.RESOLVE_VULNERABILITY: "Write code that fixes the vulnerability.",
    GitLabUnitPrimitive.REVIEW_MERGE_REQUEST: "Review new hunk and old hunk of a merge request diff.",
    GitLabUnitPrimitive.SUMMARIZE_ISSUE_DISCUSSIONS: "Summarize discussions of the issue from the comments.",
    GitLabUnitPrimitive.SUMMARIZE_MERGE_REQUEST: "Summarize merge request from the comments.",
    GitLabUnitPrimitive.SUMMARIZE_REVIEW: "Summarize open reviews in merge requests.",
    GitLabUnitPrimitive.SUMMARIZE_SUBMITTED_REVIEW: "Summarize submitted reviews of the merge request.",
    GitLabUnitPrimitive.TROUBLESHOOT_JOB: "Troubleshoot why a GitLab CI job failed and suggest a fix for it.",
    GitLabUnitPrimitive.SUMMARIZE_COMMENTS: "Summarize comments of the issue or epic.",
    GitLabUnitPrimitive.WRITE_TESTS: "Write tests for the selected code.",
}
