#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

tempdir=$(mktemp -d)
cd "$tempdir"

echo "$tempdir"

function extract_kind() {
  kind=$1
  for i in $(kubectl get "$kind" --no-headers -o custom-columns=":metadata.name"); do
    kubectl get "$kind" "$i" --show-managed-fields=false -o yaml | yq --indent 2 --prettyPrint '
      sort_keys(..) |
      del(.status) |
      del(.metadata.annotations) |
      del(.metadata.creationTimestamp) |
      del(.metadata.resourceVersion) |
      del(.metadata.namespace) |
      del(.metadata.uid) |
      del(.metadata.generation) |
      del(.spec.template.metadata.annotations) |
      del(.spec.template.metadata.creationTimestamp) |
      del(.spec.progressDeadlineSeconds) |
      del(.spec.revisionHistoryLimit)
    ' -o json | jq '.' | yq -o yaml >"${kind}-$i-deployed.yml"
  done
}

extract_kind deployment
extract_kind service
extract_kind servicemonitor
extract_kind persistentvolume
extract_kind persistentvolumeclaim
extract_kind job

yq --indent 2 --prettyPrint --split-exp '(.kind|downcase) + "-" + (.metadata.name) + "-rendered"' \
  'sort_keys(..)|... style=""' \
  "${root_dir}/manifests/fauxpilot/v2/"*.yaml

# Normalize all the yaml
for i in *.yml; do
  yq "." $i -o json | jq -Sc '.' | yq '.' -o yaml --indent 2 --prettyPrint >$i.tmp
  mv $i.tmp $i
done

for i in *-rendered.yml; do
  echo "$i"
  set -x
  diff --side-by-side $i "${i%-rendered.yml}-deployed.yml" || true
done
