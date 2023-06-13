#!/usr/bin/env bash

# This script documents the once-off setup for GKE Workload Identity
# that was carried out as part of
# https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/147

set -euo pipefail
IFS=$'\n\t'

# Staging
gcloud --project unreview-poc-390200e5 container clusters update ai-assist-test \
  --region=us-central1-c \
  --workload-pool=unreview-poc-390200e5.svc.id.goog

gcloud --project unreview-poc-390200e5 container node-pools update node-pool-n2-cpu \
  --cluster=ai-assist-test \
  --region=us-central1-c \
  --workload-metadata=GKE_METADATA

gcloud --project unreview-poc-390200e5 iam service-accounts create ai-assist-test-fauxpilot \
  --project=unreview-poc-390200e5

# Allow Access as a CloudProfiler Agent
gcloud --project unreview-poc-390200e5 projects add-iam-policy-binding unreview-poc-390200e5 \
  --member "serviceAccount:ai-assist-test-fauxpilot@unreview-poc-390200e5.iam.gserviceaccount.com" \
  --role "roles/cloudprofiler.agent"

# Allow Access as a Vertex AI User
gcloud --project unreview-poc-390200e5 projects add-iam-policy-binding unreview-poc-390200e5 \
  --member "serviceAccount:ai-assist-test-fauxpilot@unreview-poc-390200e5.iam.gserviceaccount.com" \
  --role "roles/aiplatform.user"

# Allow Access as a Monitoring viewer for the stackdriver exporter
gcloud --project unreview-poc-390200e5 projects add-iam-policy-binding unreview-poc-390200e5 \
  --member "serviceAccount:ai-assist-test-fauxpilot@unreview-poc-390200e5.iam.gserviceaccount.com" \
  --role "roles/monitoring.viewer"

# Create the Model-Gateway Kubernetes Service Account access to this service account...
gcloud --project unreview-poc-390200e5 iam service-accounts add-iam-policy-binding ai-assist-test-fauxpilot@unreview-poc-390200e5.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:unreview-poc-390200e5.svc.id.goog[fauxpilot/model-gateway-serviceaccount]"

#-------------------------------------------------------------------

# Production
gcloud --project unreview-poc-390200e5 container clusters update ai-assist \
  --region=us-central1-c \
  --workload-pool=unreview-poc-390200e5.svc.id.goog

gcloud --project unreview-poc-390200e5 container node-pools update node-pool-n2-cpu \
  --cluster=ai-assist \
  --region=us-central1-c \
  --workload-metadata=GKE_METADATA

gcloud --project unreview-poc-390200e5 iam service-accounts create ai-assist-fauxpilot \
  --project=unreview-poc-390200e5

# Allow Access as a CloudProfiler Agent
gcloud --project unreview-poc-390200e5 projects add-iam-policy-binding unreview-poc-390200e5 \
  --member "serviceAccount:ai-assist-fauxpilot@unreview-poc-390200e5.iam.gserviceaccount.com" \
  --role "roles/cloudprofiler.agent"

# Allow Access as a Vertex AI User
gcloud --project unreview-poc-390200e5 projects add-iam-policy-binding unreview-poc-390200e5 \
  --member "serviceAccount:ai-assist-fauxpilot@unreview-poc-390200e5.iam.gserviceaccount.com" \
  --role "roles/aiplatform.user"

# Allow Access as a Monitoring viewer for the stackdriver exporter
gcloud --project unreview-poc-390200e5 projects add-iam-policy-binding unreview-poc-390200e5 \
  --member "serviceAccount:ai-assist-fauxpilot@unreview-poc-390200e5.iam.gserviceaccount.com" \
  --role "roles/monitoring.viewer"

# Create the Model-Gateway Kubernetes Service Account access to this service account...
gcloud --project unreview-poc-390200e5 iam service-accounts add-iam-policy-binding ai-assist-fauxpilot@unreview-poc-390200e5.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:unreview-poc-390200e5.svc.id.goog[fauxpilot/model-gateway-serviceaccount]"
