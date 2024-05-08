import os

from google.api_core.exceptions import AlreadyExists
from google.cloud import discoveryengine_v1 as discoveryengine

if os.environ.get("INGEST_DRY_RUN") == "true":
    print("INFO: Dry Run mode. Skipped.")
    exit(0)

project_id = os.environ["GCP_PROJECT_NAME"]
data_store_id = os.environ["DATA_STORE_ID"]


def create_search_app():
    client = discoveryengine.EngineServiceClient()

    engine = discoveryengine.Engine(
        display_name=data_store_id,
        solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
        search_engine_config=discoveryengine.Engine.SearchEngineConfig(
            search_tier=discoveryengine.SearchTier.SEARCH_TIER_STANDARD,
        ),
        data_store_ids=[data_store_id],
    )

    request = discoveryengine.CreateEngineRequest(
        parent=f"projects/{project_id}/locations/global/collections/default_collection",
        engine=engine,
        engine_id=data_store_id,
    )

    operation = client.create_engine(request=request)

    print("Waiting for operation to complete...")

    response = operation.result()

    print(response)


if __name__ == "__main__":
    try:
        create_search_app()
    except AlreadyExists as ex:
        print(f"INFO: Search app already exists. Skipped. {ex}")
