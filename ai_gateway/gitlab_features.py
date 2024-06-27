from enum import Enum


# TODO: Validate that these categories exist in https://gitlab.com/gitlab-com/www-gitlab-com/raw/master/data/stages.yml.
class GitLabFeatureCategory(str, Enum):
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
class GitLabUnitPrimitive(str, Enum):
    ANALYZE_CI_JOB_FAILURE = "analyze_ci_job_failure"
    CATEGORIZE_DUO_CHAT_QUESTION = "categorize_duo_chat_question"
    CODE_SUGGESTIONS = "code_suggestions"
    DOCUMENTATION_SEARCH = "documentation_search"
    DUO_CHAT = "duo_chat"
    EXPLAIN_CODE = "explain_code"
    EXPLAIN_VULNERABILITY = "explain_vulnerability"
    FILL_IN_MERGE_REQUEST_TEMPLATE = "fill_in_merge_request_template"
    GENERATE_COMMIT_MESSAGE = "generate_commit_message"
    GENERATE_CUBE_QUERY = "generate_cube_query"
    GENERATE_ISSUE_DESCRIPTION = "generate_issue_description"
    RESOLVE_VULNERABILITY = "resolve_vulnerability"
    REVIEW_MERGE_REQUEST = "review_merge_request"
    SEMANTIC_SEARCH_ISSUE = "semantic_search_issue"
    SUMMARIZE_ISSUE_DISCUSSIONS = "summarize_issue_discussions"
    SUMMARIZE_MERGE_REQUEST = "summarize_merge_request"
    SUMMARIZE_REVIEW = "summarize_review"
    SUMMARIZE_SUBMITTED_REVIEW = "summarize_submitted_review"


class WrongUnitPrimitives(Exception):
    pass


FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS = {
    GitLabUnitPrimitive.ANALYZE_CI_JOB_FAILURE: GitLabFeatureCategory.CONTINUOUS_INTEGRATION,
    GitLabUnitPrimitive.CATEGORIZE_DUO_CHAT_QUESTION: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.DOCUMENTATION_SEARCH: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.DUO_CHAT: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.EXPLAIN_CODE: GitLabFeatureCategory.SOURCE_CODE_MANAGEMENT,
    GitLabUnitPrimitive.EXPLAIN_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
    GitLabUnitPrimitive.FILL_IN_MERGE_REQUEST_TEMPLATE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.GENERATE_COMMIT_MESSAGE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.GENERATE_CUBE_QUERY: GitLabFeatureCategory.PRODUCT_ANALYTICS_VISUALIZATION,
    GitLabUnitPrimitive.GENERATE_ISSUE_DESCRIPTION: GitLabFeatureCategory.TEAM_PLANNING,
    GitLabUnitPrimitive.RESOLVE_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
    GitLabUnitPrimitive.REVIEW_MERGE_REQUEST: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SEMANTIC_SEARCH_ISSUE: GitLabFeatureCategory.GLOBAL_SEARCH,
    GitLabUnitPrimitive.SUMMARIZE_ISSUE_DISCUSSIONS: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SUMMARIZE_MERGE_REQUEST: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SUMMARIZE_REVIEW: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SUMMARIZE_SUBMITTED_REVIEW: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
}
