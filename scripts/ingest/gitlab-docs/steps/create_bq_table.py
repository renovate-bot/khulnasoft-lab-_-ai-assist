import json
import os
from typing import Any

from google.cloud import bigquery

client = bigquery.Client(project=os.environ["GCP_PROJECT_NAME"])


def _read_jsonl() -> list[Any]:
    rows = []
    with open(os.environ["GITLAB_DOCS_JSONL_EXPORT_PATH"], "r") as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        rows.append(json.loads(json_str))

    return rows


def run():
    client.create_dataset(os.environ["BIGQUERY_DATASET_ID"], exists_ok=True)
    table_name = os.environ["BIGQUERY_TABLE_ID"]
    print("Creating table " + table_name)

    # Create the table
    # - we use the same field names as in the original data set
    table = bigquery.Table(table_name)
    table.schema = (
        bigquery.SchemaField("content", "STRING"),
        bigquery.SchemaField("metadata", "JSON"),
        bigquery.SchemaField("url", "STRING"),
    )
    result = client.create_table(table)
    print(result)

    rows_to_insert = _read_jsonl()

    t = client.get_table(table_name)
    batch_size = 1000

    for i in range(0, len(rows_to_insert), batch_size):
        errors = client.insert_rows(t, rows_to_insert[i : i + batch_size])
        if errors == []:
            print(f"Added batch_size {i}")
        else:
            raise Exception(f"Failed to insert {errors}")


if __name__ == "__main__":
    run()
