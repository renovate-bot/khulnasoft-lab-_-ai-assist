from dependency_injector import containers, providers

from ai_gateway.internal_events.client import InternalEventsClient

__all__ = [
    "ContainerInternalEvent",
]


class ContainerInternalEvent(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    client = providers.Singleton(
        InternalEventsClient,
        enabled=config.enabled,
        batch_size=config.batch_size,
        thread_count=config.thread_count,
        endpoint=config.endpoint,
        app_id=config.app_id,
        namespace=config.namespace,
    )
