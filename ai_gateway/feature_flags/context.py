from contextvars import ContextVar
from typing import List

__all__ = ["is_feature_enabled", "current_feature_flag_context"]


def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a feature is enabled.

    Args:
        feature_name: The name of the feature.

    See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/docs/feature_flags.md
    """
    enabled_feature_flags: List[str] = current_feature_flag_context.get()
    return feature_name in enabled_feature_flags


current_feature_flag_context: ContextVar[List[str]] = ContextVar(
    "current_feature_flag_context", default=[]
)
