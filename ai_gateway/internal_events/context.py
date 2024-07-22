from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

__all__ = [
    "EventContext",
    "current_event_context",
]


class EventContext(BaseModel):
    """
    This model class reperesents the available attributes in AI Gateway for the GitLab standard context.

    See https://gitlab.com/gitlab-org/iglu/-/tree/master/public/schemas/com.gitlab/gitlab_standard?ref_type=heads
    about the spec of the GitLab standard context.
    """

    environment: Optional[str] = "unknown"
    source: Optional[str] = "unknown"
    realm: Optional[str] = None
    instance_id: Optional[str] = "unknown"
    host_name: Optional[str] = "unknown"
    instance_version: Optional[str] = "unknown"
    global_user_id: Optional[str] = "unknown"


current_event_context: ContextVar[EventContext] = ContextVar(
    "current_event_context", default=EventContext()
)
