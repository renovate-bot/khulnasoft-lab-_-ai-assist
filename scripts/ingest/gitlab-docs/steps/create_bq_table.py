import json
import os
import sys
from typing import Any

from google.cloud import bigquery

# pylint: disable=direct-environment-variable-reference
if os.environ.get("INGEST_DRY_RUN") == "true":
    print("INFO: Dry Run mode. Skipped.")
    sys.exit(0)

client = bigquery.Client(project=os.environ["GCP_PROJECT_NAME"])
# pylint: enable=direct-environment-variable-reference


def _read_jsonl() -> list[Any]:
    rows = []
    # pylint: disable=direct-environment-variable-reference
    with open(os.environ["GITLAB_DOCS_JSONL_EXPORT_PATH"], "r") as json_file:
        json_list = list(json_file)
    # pylint: enable=direct-environment-variable-reference

    for json_str in json_list:
        rows.append(json.loads(json_str))

    return rows


def run():
    # pylint: disable=direct-environment-variable-reference
    client.create_dataset(os.environ["BIGQUERY_DATASET_ID"], exists_ok=True)
    table_name = os.environ["BIGQUERY_TABLE_ID"]
    # pylint: enable=direct-environment-variable-reference
    print("Creating table " + table_name)

    # Create the table
    # - we use the same field names as in the original data set
    table = bigquery.Table(table_name)
    table.schema = (
        bigquery.SchemaField("content", "STRING"),
        bigquery.SchemaField("metadata", "JSON"),
    )
    result = client.create_table(table)
    print(result)

    rows_to_insert = _read_jsonl()

    t = client.get_table(table_name)
    batch_size = 1000

    for i in range(0, len(rows_to_insert), batch_size):
        errors = client.insert_rows(t, rows_to_insert[i : i + batch_size])
        if not errors:
            print(f"Added batch_size {i}")
        else:
            raise RuntimeError(f"Failed to insert {errors}")


if __name__ == "__main__":
    run()
