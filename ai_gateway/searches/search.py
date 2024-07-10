import re
from abc import abstractmethod
from typing import Any, Dict, List

import structlog
from fastapi import HTTPException, status
from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import discoveryengine
from google.protobuf.json_format import MessageToDict

from ai_gateway.models import ModelAPIError
from ai_gateway.tracking import log_exception

SEARCH_APP_NAME = "gitlab-docs"

log = structlog.stdlib.get_logger("chat")


def _convert_version(version: str) -> str:
    # Regex to match the major and minor version numbers
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match:
        # Extract major and minor parts and join them with a hyphen
        major, minor = match.groups()
        return f"{major}-{minor}"

    raise ValueError(f"Invalid version: {version}")


def _get_data_store_id(gl_version: str) -> str:
    data_store_version = _convert_version(gl_version)

    return f"{SEARCH_APP_NAME}-{data_store_version}"


# TODO: Both Vertex Model API and Search API use the same error hierarchy under the hood via
# google-api-core (https://googleapis.dev/python/google-api-core/latest/). We would need to
# extract ModelAPIError to a common module that can be shared between /searches and /models
# module.
class VertexAPISearchError(ModelAPIError):
    @classmethod
    def from_exception(cls, ex: GoogleAPIError):
        message = f"Vertex Search API error: {type(ex).__name__}"

        if hasattr(ex, "message"):
            message = f"{message} {ex.message}"

        return cls(message, errors=(ex,))


class DataStoreNotFound(Exception):
    def __init__(self, message="", input=""):
        super().__init__(message)
        self.input = input


class Searcher:
    async def search_with_retry(self, *args, **kwargs):
        return await self.search(*args, **kwargs)

    @abstractmethod
    async def search(
        self,
        query: str,
        gl_version: str,
        page_size: int = 20,
        **kwargs: Any,
    ) -> List[Dict[Any, Any]]:
        pass

    @abstractmethod
    def provider(self):
        pass


class VertexAISearch(Searcher):
    def __init__(
        self,
        client: discoveryengine.SearchServiceAsyncClient,
        project: str,
        fallback_datastore_version: str,
        *args: Any,
        **kwargs: Any,
    ):
        self.client = client
        self.project = project
        self.fallback_datastore_version = fallback_datastore_version

    async def search_with_retry(self, *args, **kwargs):
        try:
            try:
                return await self.search(*args, **kwargs)
            except DataStoreNotFound as ex:
                log_exception(ex, extra={"input": ex.input})

            # Retry with the fallback datastore version
            kwargs["gl_version"] = self.fallback_datastore_version

            try:
                return await self.search(*args, **kwargs)
            except DataStoreNotFound as ex:
                log_exception(ex, extra={"input": ex.input})
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Data store not found.",
                )
        except VertexAPISearchError as ex:
            log_exception(ex)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vertex API Search Error.",
            )

    async def search(
        self,
        query: str,
        gl_version: str,
        page_size: int = 20,
        **kwargs: Any,
    ) -> List[Dict[Any, Any]]:
        try:
            data_store_id = _get_data_store_id(gl_version)
        except ValueError as ex:
            raise DataStoreNotFound(str(ex), input=gl_version)

        # The full resource name of the searches engine serving config
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
            page_size=page_size,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
            **kwargs,
        )

        try:
            response = await self.client.search(request)
        except NotFound:
            raise DataStoreNotFound("Data store not found", input=data_store_id)
        except GoogleAPIError as ex:
            raise VertexAPISearchError.from_exception(ex)

        return self._parse_response(MessageToDict(response._pb))

    def provider(self):
        return "vertex-ai"

    def _parse_response(self, response):
        results = []

        if "results" in response:
            for r in response["results"]:
                search_result = {
                    "id": r["document"]["id"],
                    "content": r["document"]["structData"]["content"],
                    "metadata": r["document"]["structData"]["metadata"],
                }
                results.append(search_result)

        return results
