from enum import Enum


# TODO: Validate that these categories exist in https://gitlab.com/gitlab-com/www-gitlab-com/raw/master/data/stages.yml.
class GitLabFeatureCategory(str, Enum):
    AI_ABSTRACTION_LAYER = "ai_abstraction_layer"
    CODE_REVIEW_WORKFLOW = "code_review_workflow"
    CODE_SUGGESTIONS = "code_suggestions"
    CONTINUOUS_INTEGRATION = "continuous_integration"
    DUO_CHAT = "duo_chat"
    PRODUCT_ANALYTICS_VISUALIZATION = "product_analytics_visualization"
    VULNERABILITY_MANAGEMENT = "vulnerability_management"


# Make sure these unit primitives are defined in `ee/config/cloud_connector/access_data.yml`
class GitLabUnitPrimitive(str, Enum):
    ANALYZE_CI_JOB_FAILURE = "analyze_ci_job_failure"
    ANTHROPIC_PROXY = "anthropic_proxy"
    CODE_SUGGESTIONS = "code_suggestions"
    DOCUMENTATION_SEARCH = "documentation_search"
    DUO_CHAT = "duo_chat"
    EXPLAIN_CODE = "explain_code"
    EXPLAIN_VULNERABILITY = "explain_vulnerability"
    FILL_IN_MERGE_REQUEST_TEMPLATE = "fill_in_merge_request_template"
    GENERATE_COMMIT_MESSAGE = "generate_commit_message"
    GENERATE_CUBE_QUERY = "generate_cube_query"
    GENERATE_DESCRIPTION = "generate_description"
    RESOLVE_VULNERABILITY = "resolve_vulnerability"
    REVIEW_MERGE_REQUEST = "review_merge_request"
    SUMMARIZE_COMMENTS = "summarize_comments"
    SUMMARIZE_NEW_MERGE_REQUEST = "summarize_new_merge_request"
    SUMMARIZE_REVIEW = "summarize_review"
    SUMMARIZE_SUBMITTED_REVIEW = "summarize_submitted_review"
    VERTEX_AI_PROXY = "vertex_ai_proxy"


FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS = {
    GitLabUnitPrimitive.ANALYZE_CI_JOB_FAILURE: GitLabFeatureCategory.CONTINUOUS_INTEGRATION,
    GitLabUnitPrimitive.DOCUMENTATION_SEARCH: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.DUO_CHAT: GitLabFeatureCategory.DUO_CHAT,
    GitLabUnitPrimitive.EXPLAIN_CODE: GitLabFeatureCategory.AI_ABSTRACTION_LAYER,
    GitLabUnitPrimitive.EXPLAIN_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
    GitLabUnitPrimitive.FILL_IN_MERGE_REQUEST_TEMPLATE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.GENERATE_COMMIT_MESSAGE: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.GENERATE_CUBE_QUERY: GitLabFeatureCategory.PRODUCT_ANALYTICS_VISUALIZATION,
    GitLabUnitPrimitive.GENERATE_DESCRIPTION: GitLabFeatureCategory.AI_ABSTRACTION_LAYER,
    GitLabUnitPrimitive.RESOLVE_VULNERABILITY: GitLabFeatureCategory.VULNERABILITY_MANAGEMENT,
    GitLabUnitPrimitive.REVIEW_MERGE_REQUEST: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SUMMARIZE_COMMENTS: GitLabFeatureCategory.AI_ABSTRACTION_LAYER,
    GitLabUnitPrimitive.SUMMARIZE_NEW_MERGE_REQUEST: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
    GitLabUnitPrimitive.SUMMARIZE_REVIEW: GitLabFeatureCategory.AI_ABSTRACTION_LAYER,
    GitLabUnitPrimitive.SUMMARIZE_SUBMITTED_REVIEW: GitLabFeatureCategory.CODE_REVIEW_WORKFLOW,
}
