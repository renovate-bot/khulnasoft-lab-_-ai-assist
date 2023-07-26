from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

from snowplow_tracker import AsyncEmitter, SelfDescribingJson, StructuredEvent, Tracker

__all__ = [
    "SnowplowClient",
    "SnowplowClientConfiguration",
    "SnowplowClientStub",
    "SnowplowEvent",
    "SnowplowEventContext",
]


class SnowplowClientConfiguration(NamedTuple):
    """Store all Snowplow configuration."""

    endpoint: str
    namespace: str = "gl"
    app_id: str = "gitlab_ai_gateway"


class SnowplowEventContext(NamedTuple):
    """Additional context that attached to SnowplowEvent."""

    suggestions_shown: int
    suggestions_failed: int
    suggestions_accepted: int
    prefix_length: int
    suffix_length: int
    language: str
    user_agent: str
    gitlab_realm: str


class SnowplowEvent(NamedTuple):
    """Abstracted Snowplow event."""

    context: Optional[SnowplowEventContext] = None
    category: str = "code_suggestions"
    action: str = "suggestions_requested"


class Client(ABC):
    @abstractmethod
    def track(self, *args, **kwargs) -> None:
        pass


class SnowplowClient(Client):
    """The Snowplow client to send tracking event to external Snowplow collectors.

    Attributes:
        configuration: A SnowplowClientConfiguration using to initialize the Snowplow tracker.
    """

    SCHEMA = "iglu:com.gitlab/code_suggestions_context/jsonschema/1-0-0"

    def __init__(self, configuration: SnowplowClientConfiguration) -> None:
        e = AsyncEmitter(
            batch_size=1,
            thread_count=5,
            endpoint=configuration.endpoint,
        )

        self.tracker = Tracker(
            app_id=configuration.app_id, namespace=configuration.namespace, emitters=[e]
        )

    def track(self, event: SnowplowEvent) -> None:
        """Send event to Snowplow.

        Args:
            event: A domain event which is transformed to Snowplow StructuredEvent for tracking.
        """
        structured_event = StructuredEvent(
            context=[SelfDescribingJson(self.SCHEMA, event.context._asdict())],
            category=event.category,
            action=event.action,
        )

        self.tracker.track(structured_event)


class SnowplowClientStub(Client):
    """The stub class used when Snowplow is disabled, e.g. development and testing."""

    def track(self, event: SnowplowEvent) -> None:
        pass
