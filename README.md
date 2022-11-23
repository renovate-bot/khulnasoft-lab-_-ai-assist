
# GitLab AI Assist

This project is based on the open source project 
[FauxPilot](https://github.com/moyix/fauxpilot/blob/main/docker-compose.yaml) as an initial iteration in an effort to 
create a GitLab owned AI Assistant to help developers write secure code by the 
[AI Assist SEG](https://about.gitlab.com/handbook/engineering/incubation/ai-assist/).

It uses the [SalesForce CodeGen](https://github.com/salesforce/CodeGen) models inside of NVIDIA's 
[Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) with the 
[FasterTransformer backend](https://github.com/triton-inference-server/fastertransformer_backend/).

## Prerequisites

You'll need:

* Docker
* `docker compose` >= 1.28
* An NVIDIA GPU with Compute Capability >= 6.0 and enough VRAM to run the model you want.
* [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)
* `curl` and `zstd` for downloading and unpacking the models.

Note that the VRAM requirements listed by `setup.sh` are *total* -- if you have multiple GPUs, you can split the model 
across them. So, if you have two NVIDIA RTX 3080 GPUs, you *should* be able to run the 6B model by putting half on each 
GPU.

## Configuration

Below described the configuration per component

### API
All parameters for the API are available from `api/config/config.py` which heavily relies on environment variables. An
overview of all environment variables used and their default value, if you want to deviate you should make them
available in a `.env`

```dotenv
API_EXTERNAL_PORT=5001  # External port for the API used in docker-compose
TRITON_HOST=triton
TRITON_PORT=8001
TRITON_VERBOSITY=False
DOCS_URL=None  # To disable docs on the API endpoint
OPENAPI_URL=None  # To disable docs on the API endpoint
REDOC_URL=None  # To disable docs on the API endpoint
BYPASS_EXTERNAL_AUTH=False  # Can be used for local development to bypass the GitLab server side check
GITLAB_API_URL=https://gitlab.com/api/v4/  # Can be changed to GDK: http://127.0.0.1:3000/api/v4/
USE_LOCAL_CACHE=True  # Uses a local in-memory cache instead of Redis
```


## Local development

If you are on Apple Silicon, you will need to host Triton somewhere else as there is a dependency on Nvidia GPU and 
architecture. 

You can either run `make develop-local` or  `docker-compose -f docker-compose.dev.yaml up --build --remove-orphans` this
will run the API.

Next open VS Code, install the GitLab Workflow extension, or run that extension locally. 

In VS Code settings you will need to set `AI Assist Server` to `http://localhost:5000` ofcourse change the port if you 
are deviating from the default. Set `AI Assist engine` to `FauxPilot`. It should now work.

Since the feature is only for SaaS, you need to run GDK in SaaS mode:
```bash
export GITLAB_SIMULATE_SAAS=1
gdk restart
```

Then go to `admin/settings/general/account_and_limit` and enable `Allow use of licensed EE features`.

You also need to make sure that the group you are allowing, is actually `ultimate` as it's an `ultimate` only feature,
go to `admin/overview/groups` select `edit` on the group, set `plan` to `ultimate`.

In GDK you need to enable the feature flags:
```ruby
rails console

g = Group.find(22)  # id of root group
Feature.enable(:ai_assist_api, g)
Feature.enable(:ai_assist_flag, g)
```

This will allow the feature to actually return `{"user_is_allowed": true }`.

## Authentication

The intended use of this API is to be called from the 
[GitLab VS code extension](https://gitlab.com/gitlab-org/gitlab-vscode-extension), the extension authenticates users 
against the GitLab Rails API. However, we can not rely on the VS Extension to authorize users for AI Assist as it runs
on the client side, we need a server side check. So in order to do that, the extension passes along the user's token via
a header to the AI Assist API, this token is subsequently used to make a `GET` call to `/v4/ml/ai-assist` on behalf of
the user to verify that it can indeed use AI Assist. The response `{"user_is_allowed": bool}` will be cached for 1 hour
to not burden the Rails API with an excessive amount of calls.

Below diagram described the authentication flow in blue.

![Diagram](https://docs.google.com/drawings/d/e/2PACX-1vQyFs0-irUGf_t6imgBiVSfnMf4oh45w4QEusVvwlGZy22tyCErG7JV2IC87e7DvT7b8_Ni8V77BkUW/pub?w=1022&amp;h=390)

## Component overview

In above diagram the main components are shown.

### VS Code extension

The VS Code extension has the following functions:
1. Determine input parameters
   1. Stop sequences
   1. Gather code for the prompt
1. Send the input parameters to the AI Assist API using the OpenAI package
1. Parse results from AI Assist and present them as `inlineCompletions`

### AI Assist API

Is written in Python and uses the FastApi framework along with Uvicorn. It has the following functions
1. Provide a REST API for incoming calls on `/v1/completions`
1. Authenticate incoming requests against GitLab `/v4/ml/ai-assist` and cache the result
1. Convert the prompt into a format that can be used by Triton Inference server
1. Call the Triton Inference Server, await the result and parse it back as a response

### Triton Inference server

NVIDIA Triton™ Inference Server, is an open-source inference serving software that helps standardize model deployment 
and execution and delivers fast and scalable AI in production. See [https://developer.nvidia.com/nvidia-triton-inference-server](https://developer.nvidia.com/nvidia-triton-inference-server)

### GitLab API

The endpoint `/v4/ml/ai-assist` checks if a user meets the requirements to use AI Assist and returns a boolean.

## Deploying to the Kubernetes cluster 

To successfully deploy AI Assist to a k8s cluster, please, make sure your cluster supports NVIDIA® GPU hardware accelerators.
Below, we give a guideline tested specifically on the GKE cluster in the Applied ML group. Successful work 
on any other clusters is not guaranteed.

1. Create a GKE cluster with the following configuration:
   - gke version `1.24.5-gke.600`
   - image type `container-optimized OS with containerd.`
   - machine type `n1-standard-2` machines, 
   - autoscaling enabled `from 0 to 5` nodes
   - 1 Nvidia T4 GPU 16 GB GDDR6
   - Nvidia driver version: 510.47.03, CUDA version: 11.7

2. Install NVIDIA GPU device drivers (more [info](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers)):
   ```shell
   kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
   ```

3. Provision GCP persistence disk to store AI Assist models:
   ```shell
   gcloud compute disks create --size=500GB --zone=us-central1-c nfs-ai-assist-models-disk
   ```

4. Install [`cert-manager`](https://cert-manager.io/docs/):
   ```shell
   kubectl create namespace cert-manager
   kubectl config set-context --current --namespace cert-manager
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.10.0/cert-manager.yaml
   kubectl apply -f ./manifests/cert-manager/cluster-issuer.yaml
   ```

5. Install the Ingress [`NGINX`](https://kubernetes.github.io/ingress-nginx/) controller:
   ```shell
   kubectl create namespace nginx
   kubectl config set-context --current --namespace nginx
   helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx && helm repo update
   helm install nginx ingress-nginx/ingress-nginx
   ```

6. Create the `ai-assist` namespace and update the current context
   ```shell
   export KUBERNETES_AI_ASSIST_NAMESPACE=ai-assist
   kubectl create namespace $KUBERNETES_AI_ASSIST_NAMESPACE
   kubectl config set-context --current --namespace $KUBERNETES_AI_ASSIST_NAMESPACE
   ```

7. Create the `docker-registry` secret to pull private images from GitLab AI Assist registry:
   ```shell
   export DEPLOY_TOKEN_USERNAME=<USERNAME>
   export DEPLOY_TOKEN_PASSWORD=<PASSWORD>
   kubectl create secret docker-registry gitlab-registry \
      --docker-server="registry.gitlab.com" \
      --docker-username="$DEPLOY_TOKEN_USERNAME" \
      --docker-password="$DEPLOY_TOKEN_PASSWORD"   
   ```

8. Deploy NFS server and model persistence volume:
   ```shell
   kubectl apply -f ./manifests/model-nfs-server.yaml
   kubectl apply -f ./manifests/model-persistense-volumes.yaml
   ```

9. Run the k8s job to fetch the `codegen-16B-multi` model from Hugging Face:
   ```shell
   kubectl apply -f ./manifests/model-loader.yaml
   kubectl wait --for=condition=complete --timeout=30m job/model-loader-job
   ```

10. Deploy Triton Inference server including API service:
    ```shell
    kubectl apply -f ./manifests/model-serving.yaml
    ```

11. Deploy the NGINX ingress resource with TLS enabled:
   ```shell
   kubectl apply -f ./manifests/ingress/ingress-nginx.yaml
   ```

## Monitoring

Proper observability is a corner stone of a well engineered, production worthy system. So like any such system we have monitoring too.

### Prerequisites

You will need `kubectl` installed on your computer and a modicum of comfort with the command line. Following that, you will need to configure a context to connect to the cluster from which you wish to deploy, access, or remove the monitoring stack.

Additionally you will also need to have the latest version of helm and make installed.

### Deploying Monitoring

Deploying monitoring to your kubernetes cluster is rather straight forward. While connected to the cluster run the following make commands.

```shell
make monitoring-setup # This only needs to be run once
make monitoring-deploy
```

### Connecting to Grafana and Prometheus UI

Firstly let's check our context and then switch to the correct context as needed.

```shell
# Checking your context
kubectl config current-context

# Finding available contexts
kubectl config get-contexts

# Switching contexts
kubectl config use-context {{CONTEXT_NAME_HERE}}
```

#### Grafana Port Forward

Grafana is where all the dashboard of the metrics can be found. Now that we are in the correct context you can port-forward the grafana service to your local machine with the following command.

```shell
# Mapping port 80 on the service to localhost:3000 
kubectl -n monitoring port-forward service/prometheus-grafana 3000:80
```

#### Prometheus UI Port Forward

Occasionally it is useful to check which targets are being scraped by Prometheus and if they are actually receiving metrics. The following command will execute a port forward to your desired Prometheus instance, assuming you are in the correct context. You are in the correct context, right?

```shell
# Mapping port 9090 on the service to localhost:9090
kubectl -n monitoring port-forward service/prometheus-kube-prometheus-prometheus 9090:9090
```

#### Service Discovery

Once port forwarding is set up, Service Discovery is useful to determine if Prometheus can even see the ServiceMonitor you have deployed. If your ServiceMonitor does not appear after 60 seconds then it is likely to be misconfigured.

The most common things to forget when configuring a Service and ServiceMonitor are as follows:

- The label `release: prometheus` in the ServiceMonitor
- The name of the port which should be specified in the Service as well as in the ServiceMonitor (`port: web` in the ServiceMonitor and `name:web` in the Service in the below example
- The selector that should be set in `metadata.labels` in the Service and in `spec.selector.matchLabels` in the ServiceMonitor (`app: clickhouse-exploration-go` in the below example) .

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: clickhouse-exploration-go-servicemonitor
  labels:
    release: prometheus
spec:
  endpoints:
  - interval: 30s
    port: web
  selector:
    matchLabels:
      app: clickhouse-exploration-go
---
apiVersion: v1
kind: Service
metadata:
  name: clickhouse-exploration-go-service
  labels:
    app: clickhouse-exploration-go
spec:
  selector:
    app: clickhouse-exploration-go
  ports:
  - protocol: "TCP"
    name: web
    port: 4444
    targetPort: 4444
  type: LoadBalancer

```

Service Discovery only tells you if Prometheus can see the ServiceMonitor, in order to determine if the metrics can be pulled from the service you need to check the targets page.

#### Targets

Once port forwarding is setup, Targets can be found under the Status tab or one can simply go to this endpoint: <http://localhost:9090/targets?search>=

Targets are useful for debugging. Below are examples of two ServiceMonitors, connecting to a golang and python service respectively, that are not configured for scraping and do not expose a `/metrics` endpoint.

![README.PrometheusUI](./docs/assets/README.PrometheusUI.png)
