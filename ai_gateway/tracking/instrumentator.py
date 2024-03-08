from ai_gateway.tracking import Client, SnowplowEvent

__all__ = ["SnowplowInstrumentator"]


class SnowplowInstrumentator:
    def __init__(self, client: Client) -> None:
        self.client = client

    def watch(
        self,
        snowplow_event: SnowplowEvent,
    ) -> None:
        self.client.track(snowplow_event)
