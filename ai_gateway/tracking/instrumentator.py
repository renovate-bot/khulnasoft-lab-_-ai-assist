import structlog

from ai_gateway.instrumentators.base import Telemetry
from ai_gateway.tracking import (
    Client,
    RequestCount,
    SnowplowEvent,
    SnowplowEventContext,
)

__all__ = ["SnowplowInstrumentator"]

telemetry_logger = structlog.stdlib.get_logger("telemetry")


class SnowplowInstrumentator:
    SAFE_PARSE_ID_MAX_LENGTH = 12

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
        gitlab_saas_namespace_ids: str,
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
                gitlab_saas_namespace_ids=self._safe_parse_ids(
                    gitlab_saas_namespace_ids
                ),
            )
        )

        self.client.track(snowplow_event)

    def _safe_parse_ids(self, ids: str) -> list[int]:
        parsed_ids = []

        try:
            for id in ids.split(","):
                if len(id) > self.SAFE_PARSE_ID_MAX_LENGTH:
                    raise ValueError("ID can't exceed 999,999,999,999")

                parsed_ids.append(int(id))
        except ValueError as e:
            telemetry_logger.warning(f"Failed to parse IDs: {e}")

        return parsed_ids
