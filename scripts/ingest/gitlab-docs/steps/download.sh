#!/usr/bin/env bash

set -eu

rm -Rf "${GITLAB_DOCS_CLONE_DIR}"
git clone --branch "${GITLAB_DOCS_REPO_REF}" --depth 1 "${GITLAB_DOCS_REPO}" "${GITLAB_DOCS_CLONE_DIR}"
