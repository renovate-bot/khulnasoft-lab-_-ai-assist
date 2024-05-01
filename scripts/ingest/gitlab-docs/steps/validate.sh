#!/usr/bin/env bash

set -eu

echo "GITLAB_DOCS_REPO: ${GITLAB_DOCS_REPO}"

if [ -z "$GITLAB_DOCS_REPO" ]; then
    echo "GITLAB_DOCS_REPO is empty!"
    exit 1
fi

echo "GITLAB_DOCS_REPO_REF: ${GITLAB_DOCS_REPO_REF}"

if [ -z "$GITLAB_DOCS_REPO_REF" ]; then
    echo "GITLAB_DOCS_REPO_REF is empty!"
    exit 1
fi

echo "GITLAB_DOCS_CLONE_DIR: ${GITLAB_DOCS_CLONE_DIR}"

if [ -z "$GITLAB_DOCS_CLONE_DIR" ]; then
    echo "GITLAB_DOCS_CLONE_DIR is empty!"
    exit 1
fi

echo "GITLAB_DOCS_JSONL_EXPORT_PATH: ${GITLAB_DOCS_JSONL_EXPORT_PATH}"

if [ -z "$GITLAB_DOCS_JSONL_EXPORT_PATH" ]; then
    echo "GITLAB_DOCS_JSONL_EXPORT_PATH is empty!"
    exit 1
fi

echo "GCP_PROJECT_NAME: ${GCP_PROJECT_NAME}"

if [ -z "$GCP_PROJECT_NAME" ]; then
    echo "GCP_PROJECT_NAME is empty!"
    exit 1
fi

echo "SEARCH_APP_NAME: ${SEARCH_APP_NAME}"

if [ -z "$SEARCH_APP_NAME" ]; then
    echo "SEARCH_APP_NAME is empty!"
    exit 1
fi
