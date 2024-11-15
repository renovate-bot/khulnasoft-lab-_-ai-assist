from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

from snowplow_tracker import AsyncEmitter, SelfDescribingJson, StructuredEvent, Tracker

__all__ = [
    "Client",
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
    batch_size: int = 10
    thread_count: int = 1


@dataclass
class SnowplowEventContext:
    """Additional context attached to SnowplowEvent."""

    gitlab_realm: str
    gitlab_host_name: Optional[str] = None
    gitlab_instance_id: Optional[str] = None
    gitlab_instance_version: Optional[str] = None
    gitlab_global_user_id: Optional[str] = None
    gitlab_saas_duo_pro_namespace_ids: Optional[list[int]] = None
    language: Optional[str] = None
    model_engine: Optional[str] = None
    model_name: Optional[str] = None
    prefix_length: Optional[int] = None
    suffix_length: Optional[int] = None
    suggestion_source: Optional[str] = None
    api_status_code: Optional[int] = None
    debounce_interval: Optional[int] = None
    is_streaming: Optional[bool] = None
    is_invoked: Optional[bool] = None
    options_count: Optional[int] = None
    accepted_option: Optional[int] = None
    has_advanced_context: Optional[bool] = None
    is_direct_connection: Optional[bool] = None
    total_context_size_bytes: Optional[int] = None
    content_above_cursor_size_bytes: Optional[int] = None
    content_below_cursor_size_bytes: Optional[int] = None
    context_items: Optional[list[str]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    context_tokens_sent: Optional[int] = None
    context_tokens_used: Optional[int] = None
    region: Optional[str] = None


@dataclass
class SnowplowEvent:
    """Abstracted Snowplow event."""

    context: Optional[SnowplowEventContext] = None
    category: str = "code_suggestions"
    action: str = "suggestion_requested"
    label: Optional[str] = None
    value: Optional[int] = None


class Client(ABC):
    @abstractmethod
    def track(self, *args, **kwargs) -> None:
        pass


class SnowplowClient(Client):
    """The Snowplow client to send tracking event to external Snowplow collectors.

    Attributes:
        configuration: A SnowplowClientConfiguration using to initialize the Snowplow tracker.
    """

    SCHEMA = "iglu:com.gitlab/code_suggestions_context/jsonschema/3-6-0"

    def __init__(self, configuration: SnowplowClientConfiguration) -> None:
        emitter = AsyncEmitter(
            batch_size=configuration.batch_size,
            thread_count=configuration.thread_count,
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
            context=(
                [SelfDescribingJson(self.SCHEMA, asdict(event.context))]
                if event.context
                else None
            ),
            category=event.category,
            action=event.action,
            label=event.label,
            value=event.value,
        )

        self.tracker.track(structured_event)


class SnowplowClientStub(Client):
    """The stub class used when Snowplow is disabled, e.g. development and testing."""

    def track(self, event: SnowplowEvent) -> None:
        pass
