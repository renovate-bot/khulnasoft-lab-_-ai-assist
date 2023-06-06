#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

function usage() {
    cat <<EOF 1>&2
Usage:
  scripts/helm-deploy.sh [environment] [action] --no-dry-run

  environment:
    the cluster to operate on needs to be one of [gstg, gprd]

  action: the action to perform, needs to be one off [upgrade, diff]

    diff: will print out the current changes in comparison with the specified cluster
    upgrade: will deploy a new release to the cluster with the current changes

  --no-dry-run: can only be used with the upgrade command

    By default upgrade will only print out the changes, specifying --no-dry-run will actually apply them
EOF
}

function fail() {
    echo "Error: $1" 1>&2
    echo
    usage
    exit 1
}


export HELM_DIFF_COLOR=true

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GCP_PROJECT="unreview-poc-390200e5"
GCP_ZONE="us-central1-c"

GSTG_KUBE_CTX="gke_unreview-poc-390200e5_us-central1-c_ai-assist-test"
GPRD_KUBE_CTX="gke_unreview-poc-390200e5_us-central1-c_ai-assist"
GSTG_VALUES="$INFRA_DIR/environment/test/values.yaml"
GPRD_VALUES="$INFRA_DIR/ai-assist/values.yaml"
GSTG_CLUSTER_NAME="ai-assist-test"
GPRD_CLUSTER_NAME="ai-assist"

if ! hash helm > /dev/null; then
    fail "helm is required"
fi
HELM_PLUGINS=$(helm plugin ls)
if [[ "$HELM_PLUGINS" != *"diff"* ]]; then
    fail "helm-diff plugin for helm is required"
fi

DEPLOY_ENV=$1
case $DEPLOY_ENV in
    gprd)
        KUBE_CTX=$GPRD_KUBE_CTX
        VALUES_FILE=$GPRD_VALUES
        CLUSTER_NAME=$GPRD_CLUSTER_NAME
        ;;
    gstg)
        KUBE_CTX=$GSTG_KUBE_CTX
        VALUES_FILE=$GSTG_VALUES
        CLUSTER_NAME=$GSTG_CLUSTER_NAME
        ;;
    *)
        fail "Deploy env needs to one of [gprd, gstg], given $DEPLOY_ENV"
        ;;
esac

COMMAND=$2
case $COMMAND in
    init)
        gcloud container clusters get-credentials $CLUSTER_NAME --zone $GCP_ZONE --project $GCP_PROJECT
        ;;
    diff)
        HELM_CMD="helm diff upgrade --kube-context $KUBE_CTX ai-assist ai-assist -n fauxpilot -f $VALUES_FILE"
        echo "> $HELM_CMD"
        eval "${HELM_CMD}"
        ;;
    upgrade)
        DRY_RUN_OPT="--dry-run"
        if [ -n "${3+x}" ]; then
            if [ "$3" == "--no-dry-run" ]; then
                DRY_RUN_OPT=""
            else
                fail "only '--no-dry-run' supported as 3rd argument inconjunction with upgrade"
            fi
        fi

        HELM_CMD="helm upgrade --kube-context $KUBE_CTX ai-assist ai-assist -n fauxpilot -f $VALUES_FILE $DRY_RUN_OPT"
        echo "> $HELM_CMD"
        eval "${HELM_CMD}"
        ;;
    *)
        fail "only diff and upgrade commands are allowed"
        ;;
esac
