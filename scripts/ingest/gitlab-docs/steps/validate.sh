#!/usr/bin/env bash

set -e

# -- echo helpers
RED='\033[0;31m'
NC='\033[0m'        # no color

red() {
  declare arg1="$1"
  echo -e "${RED}$arg1${NC}"
}
# /- echo helpers


fail=0

if [ -z "$GITLAB_DOCS_REPO" ]; then
    red "GITLAB_DOCS_REPO is empty!"
    fail=1
else
    echo "GITLAB_DOCS_REPO: ${GITLAB_DOCS_REPO}"
fi


if [ -z "$GITLAB_DOCS_REPO_REF" ]; then
    red "GITLAB_DOCS_REPO_REF is empty!"
    fail=1
else
    echo "GITLAB_DOCS_REPO_REF: ${GITLAB_DOCS_REPO_REF}"
fi

if [ -z "$GITLAB_DOCS_CLONE_DIR" ]; then
    red "GITLAB_DOCS_CLONE_DIR is empty!"
    fail=1
else
    echo "GITLAB_DOCS_CLONE_DIR: ${GITLAB_DOCS_CLONE_DIR}"
fi

if [ -z "$GITLAB_DOCS_JSONL_EXPORT_PATH" ]; then
    red "GITLAB_DOCS_JSONL_EXPORT_PATH is empty!"
    fail=1
else
    echo "GITLAB_DOCS_JSONL_EXPORT_PATH: ${GITLAB_DOCS_JSONL_EXPORT_PATH}"
fi

if [ -z "$GCP_PROJECT_NAME" ]; then
    red "GCP_PROJECT_NAME is empty!"
    fail=1
else
    echo "GCP_PROJECT_NAME: ${GCP_PROJECT_NAME}"
fi

if [ -z "$SEARCH_APP_NAME" ]; then
    red "SEARCH_APP_NAME is empty!"
    fail=1
else
    echo "SEARCH_APP_NAME: ${SEARCH_APP_NAME}"
fi

if [ $fail == 1 ]; then
    echo
    red "Validation failed."
    exit 1
fi
