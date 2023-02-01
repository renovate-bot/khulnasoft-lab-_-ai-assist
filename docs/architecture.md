# Architecture overview 

This file provides an overview of the Code Suggestions (CS) architecture post 
Milestone ["Code Suggestions Gated MVC"](https://gitlab.com/groups/gitlab-org/modelops/applied-ml/code-suggestions/-/epics/2).
Please, open new MRs in case of any errors, typos or to actualize the information.

Code Suggestions components:
- GitLab VSCode extension
- Model Gateway (the current repo)
- Salesforce Codegen model

We rely on Google Kubernetes Engine (GKE) with the GPU support enabled to deploy the above components.
Please note that Code Suggestions currently provides code completions only in Visual Studio Code with the 
official GitLab extension installed.

## Working with Code Suggestions in VSCode
This GitLab VSCode extension is responsible for code autocompletion at a given user's prompt. Depending on the prompt,
the extension either provides entire code snippets (e.g., generating functions) or completes the current line 
(e.g., completing comments, function calls). The extension provides code completions in a target file without 
analyzing file extensions and other files in the repo. At this stage, the extension takes the context above the cursor 
and send it to `model-gateway` for generating code completions. Please note that further changes in the extension related 
to analyzing project files and their extensions will require changes in the used model. You can find more details on 
how to enable and configure the Gitlab VSCode extension in TBD.

## Wrapping business logic in Model Gateway
Model Gateway is responsible for wrapping the Codegen model with business logic. 
After receiving requests at the endpoint `/v1/completions`, Model Gateway performs the following steps:
- authenticate users by calling `gitlab.com/api/v4/ml/ai-assist` with the PAT sent along with the incoming request
- call the Codegen model using the gRPC Triton client stub to get code completions 
- postprocess the generated code completion to mask IPv4/v6, email addresses, and secrets

To optimise the number of calls to the GitLab core for user authentication, Model Gateway caches successful 
authentication responses for 1 hour. So, if a user uses Code Suggestions for 3 hours of continuous coding, 
Model Gateway will make 2 authentication requests. At this stage, we use the in-memory store to cache authentication 
responses as we try to introduce new features iteratively and simplify the infrastructure readiness review. As the load 
increases, we might consider using Redis or Postgres as a cache store depending on the product decision and GitLab requirements.

We're able to detect and mask the following secrets:
- basic auth, e.g., `git clone https://username:password@gitlab.com/username/repository.git`
- artifactory credentials
- sendgrid tokens
- azure storage tokens
- discord tokens
- twilio tokens
- secret-sounding variable names. Here are the use cases we support - https://github.com/Yelp/detect-secrets/blob/master/tests/plugins/keyword_test.py#L126

We strive to follow the [Clean Architecture pattern](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
when developing Model Gateway. We moved authentication and logging to their dedicated middlewares:
`MiddlewareLogRequest` and `MiddlewareAuthentication`. Model-specific and business logic have been separated into 
dedicated layers as well: `models.Codegen` and `suggestions.CodeSuggestionsUseCase`.

## Generating code completions with CodeGen
As the base model, we use Salesforce Codegen Large-Language Autoregressive model with 16B parameters. The Codegen models supports the following languages:
C, C++, Go, Java, JavaScript, and Python. To serve this model, we rely on Triton Server by Nvidia with the fastertransformer (FT)
backend enabled. Triton optimizes serving for real-time inferencing under strict latency constraints with dynamic batching,
supports batch inferencing to maximize GPU and CPU utilization. Triton provides multiple backends to serve different types of models. 
The FT backend can be considered as an add-on for Triton to efficiently serve large transformer models.

We can describe the current model iteration (`/models/fauxpilot`) as an ensemble of:
- **Encoder served by the Python Triton backend.** We use the official Codegen encoder model released via HuggingFace.
- **Converted Codegen (FauxPilot) model served by the FT Triton backend.** The FT backend doesn't support the
  architecture of the original Codegen model. To overcome this and still use Triton, we deploy the FauxPilot model, 
  which is the version of Codegen converted by Brendan Dolan-Gavitt to GPT-J.
- **Decoder served by the Python Triton backend.** The decoder represents the same model as the encoder, 
  but is called in reverse to convert identifiers to meaningful tokens. 

Autoregressive models use a variety of next-token decision strategies during code generation. According to the latest research, 
contrastive search is SOTA. In our setting, due to the lack of contrastive search implementation in Triton, 
we have settled on using the `top-p` sampling following the original Codegen paper. Please, find other main hyperparameters
used by the ensemble model:
```
# Number of tokens to generate
REQUEST_OUTPUT_LEN = 16

# Model hyperparameters
MODEL_TEMPERATURE = .2
MODEL_REPETITION_PENALTY = 1
MODEL_TOP_K = 0
MODEL_TOP_P = .95
MODEL_PAD_ID = 50256
```

Autoregressive models require a stop criterion to avoid infinite generation. In our setting, 
we use the maximum number of new generated tokens `REQUEST_OUTPUT_LEN` as a stop criterion. 
We can explore other criteria depending on the product direction.
