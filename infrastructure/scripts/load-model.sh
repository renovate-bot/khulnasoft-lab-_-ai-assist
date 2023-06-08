#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
manifest_file=${root_dir}/manifests/fauxpilot/model-loader-job/model-loader.yaml

echo "----------------------------"
echo "Using context $(kubectl config current-context)"
echo "----------------------------"

echo "----------------------------"
echo "▶️  Running 'kubectl create -f ${manifest_file}'"
job_json=$(kubectl create -f "$manifest_file" -o=json)
job_name=$(echo "$job_json" | jq -r '.metadata.name')
kube_path=jobs.batch/$job_name

pod=""
until [[ $pod != "" ]]; do
    echo "Waiting for a pod from $job_name to be running"
    sleep 10
    # get only one pod from job, not many
    pod=$(kubectl get pods --selector=job-name="${job_name}" --field-selector=status.phase!=Pending --output=jsonpath='{.items[*].metadata.name}' | cut -d ' ' -f1)
done

kubectl logs "$kube_path" --all-containers --follow
