from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

__all__ = [
    "EventContext",
    "current_event_context",
    "tracked_internal_events",
    "InternalEventAdditionalProperties",
]


class EventContext(BaseModel):
    """
    This model class represents the available attributes in AI Gateway for the GitLab standard context.

    See https://gitlab.com/gitlab-org/iglu/-/tree/master/public/schemas/com.gitlab/gitlab_standard?ref_type=heads
    about the spec of the GitLab standard context.
    """

    environment: Optional[str] = "development"
    source: Optional[str] = "ai-gateway-python"
    realm: Optional[str] = None
    instance_id: Optional[str] = None
    host_name: Optional[str] = None
    instance_version: Optional[str] = None
    global_user_id: Optional[str] = None
    context_generated_at: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    is_gitlab_team_member: Optional[bool] = None
    feature_enabled_by_namespace_ids: Optional[List[int]] = None
    project_id: Optional[int] = None
    namespace_id: Optional[int] = None
    plan: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class InternalEventAdditionalProperties:
    """Internal event additional properties.

    Attributes:
        label: Label of the event. It's recommended to set an accessed unit primitive name for the event.
        property: Property for the event, representing a specific attribute.
        value: Numeric value associated with the event.
        extra: Additional key-value pairs that cannot be added to the standard context, label, or property.
               For Example `model_engine`, `prefix_length`, etc.
    """

    label: Optional[str] = None
    property: Optional[str] = None
    value: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        self.label = kwargs.pop("label", None)
        self.property = kwargs.pop("property", None)
        self.value = kwargs.pop("value", None)
        self.extra = kwargs


current_event_context: ContextVar[EventContext] = ContextVar(
    "current_event_context", default=EventContext()
)

tracked_internal_events: ContextVar[Set[str]] = ContextVar(
    "tracked_internal_events", default=set()
)
