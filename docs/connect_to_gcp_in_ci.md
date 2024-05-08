# Connect to GCP in CI/CD pipelines

Some jobs in CI/CD pipelines in AI Gateway project connect to GCP for managing resources, such as [`ingest:dev`](./search.md).
We use OpenID Connect with GCP Workload Identity Federation for authentication and authorization to GCP resources.

## Setup

Here is how to setup service account, workload identity pool/provider and IAM permissions in a GCP project:

Create a service account:

```shell
SERVICE_ACCOUNT_NAME="ai-gateway-sa-example"
PROJECT_ID="<gcp-project-id>"

gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
  --project "${PROJECT_ID}"
```

Create a workload identity pool:

```shell
POOL_ID="gitlab-ai-gateway"
POOL_DESCRIPTION="AI Gateway"

gcloud iam workload-identity-pools create "${POOL_ID}" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --display-name="${POOL_DESCRIPTION}"
```

Create a provider in the workload identity pool:

```shell
PROVIDER_ID="gitlab-ai-gateway-provider"
PROVIDER_DESCRIPTION="AI Gateway provider"

gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="${POOL_ID}" \
  --display-name="${PROVIDER_DESCRIPTION}" \
  --attribute-mapping="google.subject=assertion.sub,attribute.project_path=assertion.project_path,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.ref == 'main'" \
  --issuer-uri="https://gitlab.com/" \
  --allowed-audiences="https://gitlab.com"
```

Allow an external user to impersonate the service account:

```shell
WORKLOAD_IDENTITY_POOL_ID=$(gcloud iam workload-identity-pools describe "${POOL_ID}" --project="${PROJECT_ID}" --location="global" --format="value(name)")
GITLAB_PROJECT_PATH="gitlab-org/modelops/applied-ml/code-suggestions/ai-assist"

gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_ID}/attribute.project_path/${GITLAB_PROJECT_PATH}"
```

Allow the service account to access a resource. In this case, we grant BigQuery and Discovery Engine admin permissions:

```shell
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/discoveryengine.admin"
```

Get the workload identity provider ID. This value will be specified in `WORKLOAD_IDENTITY_PROVIDER` in `.gitlab-ci.yml`:

```shell
gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="${POOL_ID}" \
  --format="value(name)"
```

Get the service account name. This value will be specified in `SERVICE_ACCOUNT` in `.gitlab-ci.yml`:

```shell
echo "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
```

## .gitlab-ci.yml

You can use the service account created in the previous section in `.gitlab-ci.yml`:

```yaml
job:
  id_tokens:
    GITLAB_OIDC_TOKEN:
      aud: https://gitlab.com
  variables:
    WORKLOAD_IDENTITY_PROVIDER: <fetched-above>
    SERVICE_ACCOUNT: <fetched-above>
    GCP_PROJECT_NAME: <gcp-project-id>
  before_script:
    - echo ${GITLAB_OIDC_TOKEN} > .ci_job_jwt_file
    - gcloud iam workload-identity-pools create-cred-config "${WORKLOAD_IDENTITY_PROVIDER}"
      --service-account="${SERVICE_ACCOUNT}" --output-file=.gcp_temp_cred.json
      --credential-source-file=`pwd`/.ci_job_jwt_file
    - gcloud auth login --cred-file=`pwd`/.gcp_temp_cred.json
    - gcloud config set project ${GCP_PROJECT_NAME}
    - export GOOGLE_APPLICATION_CREDENTIALS=`pwd`/.gcp_temp_cred.json
```

See `ingest.gitlab-ci.yml` for example.

## Reference

- [Configure OpenID Connect with GCP Workload Identity Federation](https://docs.gitlab.com/ee/ci/secrets/id_token_authentication.html)
- [Connect to cloud services](https://docs.gitlab.com/ee/ci/cloud_services/index.html)
- [OpenID Connect (OIDC) Authentication Using ID Tokens](https://docs.gitlab.com/ee/ci/secrets/id_token_authentication.html)
