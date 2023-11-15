from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

from snowplow_tracker import AsyncEmitter, SelfDescribingJson, StructuredEvent, Tracker

__all__ = [
    "Client",
    "RequestCount",
    "SnowplowClient",
    "SnowplowClientConfiguration",
    "SnowplowClientStub",
    "SnowplowEvent",
    "SnowplowEventContext",
]


@dataclass
class SnowplowClientConfiguration:
    """Store all Snowplow configuration."""

    endpoint: str
    namespace: str = "gl"
    app_id: str = "gitlab_ai_gateway"


@dataclass
class RequestCount:
    """Acceptance, show and error counts for previous requests."""

    requests: int
    errors: int
    accepts: int
    lang: Optional[str]
    model_engine: Optional[str]
    model_name: Optional[str]


@dataclass
class SnowplowEventContext:
    """Additional context that attached to SnowplowEvent."""

    request_counts: Optional[list[RequestCount]]
    prefix_length: int
    suffix_length: int
    language: str
    user_agent: str
    gitlab_realm: str
    gitlab_instance_id: str
    gitlab_global_user_id: str
    gitlab_host_name: str


@dataclass
class SnowplowEvent:
    """Abstracted Snowplow event."""

    context: Optional[SnowplowEventContext] = None
    category: str = "code_suggestions"
    action: str = "suggestion_requested"


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
        emitter = AsyncEmitter(
            batch_size=1,
            thread_count=5,
            endpoint=configuration.endpoint,
        )

        self.tracker = Tracker(
            app_id=configuration.app_id,
            namespace=configuration.namespace,
            emitters=[emitter],
        )

    def track(self, event: SnowplowEvent) -> None:
        """Send event to Snowplow.

        Args:
            event: A domain event which is transformed to Snowplow StructuredEvent for tracking.
        """
        structured_event = StructuredEvent(
            context=[SelfDescribingJson(self.SCHEMA, asdict(event.context))],
            category=event.category,
            action=event.action,
        )

        self.tracker.track(structured_event)


class SnowplowClientStub(Client):
    """The stub class used when Snowplow is disabled, e.g. development and testing."""

    def track(self, event: SnowplowEvent) -> None:
        pass
