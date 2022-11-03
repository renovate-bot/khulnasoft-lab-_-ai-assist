
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

