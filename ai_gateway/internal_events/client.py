from typing import Optional

from snowplow_tracker import AsyncEmitter, SelfDescribingJson, StructuredEvent, Tracker

from ai_gateway.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
    tracked_internal_events,
)

__all__ = ["InternalEventsClient"]


class InternalEventsClient:
    """Client to handle internal events using SnowplowClient."""

    STANDARD_CONTEXT_SCHEMA = "iglu:com.gitlab/gitlab_standard/jsonschema/1-1-1"

    def __init__(
        self,
        enabled: bool,
        endpoint: str,
        app_id: str,
        namespace: str,
        batch_size: int,
        thread_count: int,
    ) -> None:
        self.enabled = enabled

        if enabled:
            emitter = AsyncEmitter(
                batch_size=batch_size,
                thread_count=thread_count,
                endpoint=endpoint,
            )

            self.snowplow_tracker = Tracker(
                app_id=app_id,
                namespace=namespace,
                emitters=[emitter],
            )

    def track_event(
        self,
        event_name: str,
        additional_properties: Optional[InternalEventAdditionalProperties] = None,
        category: Optional[str] = "default_category",
    ) -> None:
        """Send internal event to Snowplow.

        Args:
            event_name: The name of the event. It should follow
                <action>_<target_of_action>_<where/when>. Reference:
                https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html#defining-event-and-metrics
            additional_properties: Additional properties for the event.
            category:  the location where the event happened ideally classname which invoked the event.
        """
        if not self.enabled:
            return

        if additional_properties is None:
            additional_properties = InternalEventAdditionalProperties()

        context: EventContext = current_event_context.get()
        new_context = context.model_dump()
        new_context["extra"] = additional_properties.extra

        structured_event = StructuredEvent(
            context=[SelfDescribingJson(self.STANDARD_CONTEXT_SCHEMA, new_context)],
            category=category,
            action=event_name,
            label=additional_properties.label,
            value=additional_properties.value,
            property_=additional_properties.property,
        )

        self.snowplow_tracker.track(structured_event)
        tracked_internal_events.get().add(event_name)
