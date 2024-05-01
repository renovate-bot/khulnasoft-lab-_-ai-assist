#!/usr/bin/env bash

set -eu

STEPS_DIR="$(pwd)/scripts/ingest/gitlab-docs/steps"

echo "STEPS_DIR: ${STEPS_DIR}"

echo "------------------------------------------------------- Validating -------------------------------------------------------"
"${STEPS_DIR}"/validate.sh
echo "------------------------------------------------------- Downloading -------------------------------------------------------"
"${STEPS_DIR}"/download.sh
echo "------------------------------------------------------- Exporting Variables -------------------------------------------------------"
source "${STEPS_DIR}"/export_variables.sh
echo "------------------------------------------------------- Parsing -------------------------------------------------------"
"${STEPS_DIR}"/parse.rb
echo "------------------------------------------------------- Creating Bigquery table -------------------------------------------------------"
poetry run python "${STEPS_DIR}"/create_bq_table.py
echo "------------------------------------------------------- Creating or refreshing data store in Agent Builder -------------------------------------------------------"
poetry run python "${STEPS_DIR}"/create_data_store.py
echo "------------------------------------------------------- Creating search app in Agent Builder -------------------------------------------------------"
poetry run python "${STEPS_DIR}"/create_search_app.py
