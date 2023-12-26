from dependency_injector import containers, providers

from ai_gateway.config import Config
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import (
    SnowplowClient,
    SnowplowClientConfiguration,
    SnowplowClientStub,
)

__all__ = [
    "ContainerTracking",
]


def _init_snowplow_client(enabled: bool, configuration: SnowplowClientConfiguration):
    if not enabled:
        return SnowplowClientStub()

    return SnowplowClient(configuration)


class ContainerTracking(containers.DeclarativeContainer):
    config = providers.Configuration()
    config.from_dict(Config().model_dump())

    client = providers.Resource(
        _init_snowplow_client,
        enabled=config.snowplow.enabled,
        configuration=providers.Resource(
            SnowplowClientConfiguration,
            endpoint=config.snowplow.endpoint,
        ),
    )

    instrumentator = providers.Resource(
        SnowplowInstrumentator,
        client=client,
    )
