from typing import Optional

from fastapi import Request
from gitlab_cloud_connector import CloudConnectorConfig
from starlette.datastructures import CommaSeparatedStrings

from ai_gateway.api.middleware import (
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER,
)
from ai_gateway.tracking import SnowplowEventContext


def get_snowplow_code_suggestion_context(
    req: Request,
    region: str,
    prefix: Optional[str] = "",
    suffix: Optional[str] = "",
    language: Optional[str] = "",
    global_user_id: Optional[str] = "",
    suggestion_source: Optional[str] = "network",
) -> SnowplowEventContext:
    language = language.lower() if language else ""
    # gitlab-rails 16.3+ sends an X-Gitlab-Realm header
    gitlab_realm = req.headers.get(X_GITLAB_REALM_HEADER).lower()
    # older versions don't serve code suggestions, so we read this from the IDE token claim
    if not gitlab_realm and req.user and req.user.claims:
        gitlab_realm = req.user.claims.gitlab_realm.lower()

    is_direct_connection = False
    if (
        req.user
        and req.user.claims
        and req.user.claims.issuer == CloudConnectorConfig().service_name
    ):
        is_direct_connection = True

    return SnowplowEventContext(
        gitlab_realm=gitlab_realm if gitlab_realm else "",
        gitlab_host_name=req.headers.get(X_GITLAB_HOST_NAME_HEADER, ""),
        gitlab_instance_id=req.headers.get(X_GITLAB_INSTANCE_ID_HEADER, ""),
        gitlab_global_user_id=global_user_id,
        gitlab_saas_duo_pro_namespace_ids=list(
            map(
                int,
                CommaSeparatedStrings(
                    req.headers.get(X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER, "")
                ),
            )
        ),
        language=language,
        prefix_length=len(prefix),
        suffix_length=len(suffix),
        suggestion_source=suggestion_source,
        is_direct_connection=is_direct_connection,
        region=region,
    )
