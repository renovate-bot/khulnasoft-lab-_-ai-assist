from ai_gateway.instrumentators.base import Telemetry
from ai_gateway.tracking import (
    Client,
    RequestCount,
    SnowplowEvent,
    SnowplowEventContext,
)

__all__ = ["SnowplowInstrumentator"]


class SnowplowInstrumentator:
    def __init__(self, client: Client) -> None:
        self.client = client

    def watch(
        self,
        telemetry: list[Telemetry],
        prefix_length: int,
        suffix_length: int,
        language: str,
        user_agent: str,
        gitlab_realm: str,
        gitlab_instance_id: str,
        gitlab_global_user_id: str,
        gitlab_host_name: str,
        gitlab_saas_namespace_ids: list[str],
        gitlab_saas_duo_pro_namespace_ids: list[str],
    ) -> None:
        request_counts = []
        for stats in telemetry:
            request_count = RequestCount(
                requests=stats.requests,
                accepts=stats.accepts,
                errors=stats.errors,
                lang=stats.lang,
                model_engine=stats.model_engine,
                model_name=stats.model_name,
            )

            request_counts.append(request_count)

        snowplow_event = SnowplowEvent(
            context=SnowplowEventContext(
                request_counts=request_counts,
                prefix_length=prefix_length,
                suffix_length=suffix_length,
                language=language,
                user_agent=user_agent,
                gitlab_realm=gitlab_realm,
                gitlab_instance_id=gitlab_instance_id,
                gitlab_global_user_id=gitlab_global_user_id,
                gitlab_host_name=gitlab_host_name,
                gitlab_saas_namespace_ids=[int(id) for id in gitlab_saas_namespace_ids],
                gitlab_saas_duo_pro_namespace_ids=[
                    int(id) for id in gitlab_saas_duo_pro_namespace_ids
                ],
            )
        )

        self.client.track(snowplow_event)
