#!/usr/bin/env bash

set -euo pipefail

version_file="${GITLAB_DOCS_CLONE_DIR}/VERSION"
date=$(date "+%Y-%m-%d-%H-%M-%S")
export GITLAB_VERSION=$(cat "$version_file")

if [[ $GITLAB_VERSION =~ ^([0-9]+\.[0-9]+) ]]; then
    converted=$(echo "${BASH_REMATCH[1]}" | sed 's/\./-/g')
    export DATA_STORE_VERSION=${converted}
else
    echo "No match found."
    exit 1
fi

export DATA_STORE_ID="${SEARCH_APP_NAME}-${DATA_STORE_VERSION}"
export BIGQUERY_DATASET_ID=$(echo "$DATA_STORE_ID" | sed 's/-/_/g')
export BIGQUERY_TABLE_NAME="${date}"
export BIGQUERY_TABLE_ID="${GCP_PROJECT_NAME}.${BIGQUERY_DATASET_ID}.${BIGQUERY_TABLE_NAME}"

echo "GITLAB_VERSION: ${GITLAB_VERSION}"
echo "DATA_STORE_VERSION: ${DATA_STORE_VERSION}"
echo "DATA_STORE_ID: ${DATA_STORE_ID}"
echo "BIGQUERY_DATASET_ID: ${BIGQUERY_DATASET_ID}"
echo "BIGQUERY_TABLE_NAME: ${BIGQUERY_TABLE_NAME}"
echo "BIGQUERY_TABLE_ID: ${BIGQUERY_TABLE_ID}"
