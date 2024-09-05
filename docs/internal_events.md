# Internal Event Tracking

To collect product usage metrics, use [`InternalEventsClient`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/ai_gateway/internal_events/client.py) in AI Gateway.
This is a Python client for the [GitLab Internal Event Tracking](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html) system.

Previously, we were using [`SnowplowInstrumentator`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/ai_gateway/tracking/snowplow.py) for tracking Code Suggestion events, however, this instrumentator is deprecated since it's hard to extend for various events.
See [this issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/561) for migrating from `SnowplowInstrumentator` to `InternalEventsClient`.

## Trigger events

To trigger an event, call the `track_event` method of the `InternalEventsClient` object with the desired arguments:

```python
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.async_dependency_resolver get_internal_event_client

@router.post("/awesome_feature")
async def awesome_feature(
    request: Request,
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    # Send "request_awesome_feature" event to Snowplow.
    internal_event_client.track_event("request_awesome_feature")
```

Additional properties can be passed when tracking events. They can be used to save additional data related to a given event.

Snowplow has built-in properties with keys `label` (string), `property` (string), and `value` (numeric). It's recommended to use these properties first. If you need to pass more properties, you can send custom key-value pairs. For example:

```python
from ai_gateway.internal_events.context import InternalEventAdditionalProperties
...
# Send "request_awesome_feature" event to Snowplow with additional properties.
additional_properties = InternalEventAdditionalProperties(
    label="completion_event", property="property_value", value=1, key="value"
)
internal_event_client.track_event(
    event_name="request_awesome_feature",
    additional_properties=additional_properties,
)
```

There is another parameter called `category` where we capture where the event happened. We should pass the name of the class where the event happened.

```python
internal_event_client.track_event(
    f"request_{path_unit_primitive_map[chat_invokable]}",
    category=__name__,
)
```

There are various arguments you can set aside from the event name.
See [this section](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html#trigger-events) for more information.

## Test locally

1. Enable snowplow micro in GDK with [these instructions](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/local_setup_and_debugging.html#snowplow-micro).
1. Update [the application settings](application_settings.md#how-to-update-application-settings):

    ```shell
    AIGW_INTERNAL_EVENT__ENABLED=true
    AIGW_INTERNAL_EVENT__ENDPOINT=http://127.0.0.1:9091
    AIGW_INTERNAL_EVENT__BATCH_SIZE=1
    AIGW_INTERNAL_EVENT__THREAD_COUNT=1
    ```

1. Run snowplow micro with `gdk start snowplow-micro`.
1. Run AI Gateway with `poetry run ai_gateway`.

Visit [the UI dashboard](http://127.0.0.1:9091) to see the events received by snowplow micro.

## Configuration

There are various configuration options for the Internal Event Tracking.
See `AIGW_INTERNAL_EVENT` prefixed variables in the [application settings](application_settings.md#how-to-update-application-settings).

## Internal Event Middleware

Some of the fundamental event arguments are collected at `InternalEventMiddleware` and set to all events automatically.
