import re
from typing import Any

from google.cloud import discoveryengine as discoveryengine
from google.protobuf.json_format import MessageToDict

__all__ = ["VertexAISearch"]

SEARCH_APP_NAME = "gitlab-docs"


def _convert_version(version: str) -> str:
    # Regex to match the major and minor version numbers
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match:
        # Extract major and minor parts and join them with a hyphen
        major, minor = match.groups()
        return f"{major}-{minor}"
    else:
        raise ValueError(f"Invalid version: {version}")


def _get_data_store_id(gl_version: str) -> str:
    data_store_version = _convert_version(gl_version)

    return f"{SEARCH_APP_NAME}-{data_store_version}"


class VertexAISearch:
    def __init__(
        self,
        client: discoveryengine.SearchServiceClient,
        project: str,
        *args: Any,
        **kwargs: Any,
    ):
        self.client = client
        self.project = project

    def search(
        self,
        query: str,
        gl_version: str,
        **kwargs: Any,
    ) -> dict:
        data_store_id = _get_data_store_id(gl_version)

        # The full resource name of the search engine serving config
        # e.g. projects/{project_id}/locations/{location}/dataStores/{data_store_id}/servingConfigs/{serving_config_id}
        serving_config = self.client.serving_config_path(
            project=self.project,
            location="global",
            data_store=data_store_id,
            serving_config="default_config",
        )

        # Refer to the `SearchRequest` reference for all supported fields:
        # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            # content_search_spec=content_search_spec,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
            **kwargs,
        )

        response = self.client.search(request)

        return MessageToDict(response._pb)
