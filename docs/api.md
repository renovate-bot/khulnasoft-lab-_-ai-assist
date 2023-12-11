# API

## Authentication

The Code Suggestions API supports authentication via Code Suggestions access tokens.

You can use an Code Suggestions access token to authenticate with the API by passing it in the
`Authorization` header and specifying the `X-Gitlab-Authentication-Type` header.

```shell
curl --header "Authorization: Bearer <access_token>" --header "X-Gitlab-Authentication-Type: oidc" \
  "https://codesuggestions.gitlab.com/v2/completions"
```

## Code Suggestions

### Completions

Given a prompt, the service will return one suggestion. This endpoint supports
two versions of payloads.

- If `vertex-ai` model provider is selected, we uses `code-gecko@latest`.
- If `anthropic` model provider is selected, we uses `claude-instant-1.2`.

```plaintext
POST /v2/completions
POST /ai/v2/completions
POST /v2/code/completions
POST /ai/v2/code/completions
```

#### V1 Prompt

This performs some pre-processing of the content before forwarding it to the
third-party model provider.

| Attribute                           | Type   | Required | Description                                                                                                      | Example                   |
| ----------------------------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `prompt_version`                    | int    | yes      | The version of the prompt.                                                                                       | `1`                       |
| `project_path`                      | string | no       | The name of the project (max_len: **255**).                                                                      | `gitlab-orb/gitlab-shell` |
| `project_id`                        | int    | no       | The id of the project.                                                                                           | `33191677`                |
| `model_provider`                    | string | no       | The name of the model provider. Valid values are: `anthropic` and `vertex-ai`.                                   | `vertex-ai`               |
| `current_file`                      | hash   | yes      | The data of the current file.                                                                                    |                           |
| `current_file.file_name`            | string | yes      | The name of the current file (max_len: **255**).                                                                 | `README.md`               |
| `current_file.language_identifier`  | string | no       | The language identifier defined from editor (max_len: **255**). This overrides language detected from file name. | `python`                  |
| `current_file.content_above_cursor` | string | yes      | The content above cursor (max_len: **100,000**).                                                                 | `import numpy as np`      |
| `current_file.content_below_cursor` | string | yes      | The content below cursor (max_len: **100,000**).                                                                 | `def __main__:\n`         |
| `telemetry`                         | array  | no       | The list of telemetry data from previous request (max_len: **10**).                                              |                           |
| `telemetry.model_engine`            | string | no       | The model engine used for completions (max_len: **50**).                                                         | `vertex-ai`               |
| `telemetry.model_name`              | string | no       | The model name used for completions (max_len: **50**).                                                           | `code-gecko`              |
| `telemetry.lang`                    | string | no       | The language used for completions (max_len: **50**).                                                             | `python`                  |
| `telemetry.experiments`             | array  | no       | The list of experiments run from previous request.                                                               |                           |
| `telemetry.experiments.name`        | string | yes      | The experiment name.                                                                                             | `exp_truncate_suffix`     |
| `telemetry.experiments.variant`     | int    | yes      | The experiment variant.                                                                                          | `0`                       |
| `telemetry.requests`                | int    | yes      | The number of previously requested completions.                                                                  | `1`                       |
| `telemetry.accepts`                 | int    | yes      | The number of previously accepted completions.                                                                   | `1`                       |
| `telemetry.errors`                  | int    | yes      | The number of previously failed completions.                                                                     | `0`                       |

```shell
curl --request POST \
  --url 'https://codesuggestions.gitlab.com/v2/code/completions' \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "prompt_version": 1,
    "project_path": "gitlab-org/gitlab-shell",
    "project_id": 33191677,
    "current_file": {
      "file_name": "test.py",
      "content_above_cursor": "def is_even(n: int) ->",
      "content_below_cursor": ""
    }
    "telemetry": [
      {
        "model_engine": "vertex-ai",
        "model_name": "code-gecko",
        "lang": "python",
        "experiments": [
          {
            "name": "exp_truncate_suffix",
            "variant": 0
          }
        ],
        "requests": 1,
        "accepts": 1,
        "errors": 0
      }
    ]
  }'
```

Example response:

```json
{
  "id": "id",
  "model": {
    "engine": "vertex-ai",
    "name": "code-gecko",
    "lang": "python"
  },
  "experiments": [
    {
      "name": "exp_truncate_suffix",
      "variant": 0
    }
  ]
  "object": "text_completion",
  "created": 1682031100,
  "choices": [
    {
      "text": " bool:\n    return n % 2 == 0\n\n\ndef is_odd",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```

#### V2 Prompt

This accepts prebuilt `prompt` and forwards it directly to third-party provider.
This only supports `anthropic` model provider.

| Attribute                           | Type   | Required | Description                                                                    | Example                              |
| ----------------------------------- | ------ | -------- | ------------------------------------------------------------------------------ | ------------------------------------ |
| `prompt_version`                    | int    | yes      | The version of the prompt                                                      | `2`                                  |
| `project_path`                      | string | no       | The name of the project (max_len: **255**)                                     | `gitlab-orb/gitlab-shell`            |
| `project_id`                        | int    | no       | The id of the project                                                          | `33191677`                           |
| `model_provider`                    | string | no       | The name of the model provider. Valid values are: `anthropic` and `vertex-ai`. | `anthropic`                          |
| `current_file`                      | hash   | yes      | The data of the current file                                                   |                                      |
| `current_file.file_name`            | string | yes      | The name of the current file (max_len: **255**)                                | `README.md`                          |
| `current_file.content_above_cursor` | string | yes      | The content above cursor (max_len: **100,000**)                                | `import numpy as np`                 |
| `current_file.content_below_cursor` | string | yes      | The content below cursor (max_len: **100,000**)                                | `def __main__:\n`                    |
| `prompt`                            | string | yes      | The content of a prebuilt prompt                                               | `Human: You are a code assistant...` |
| `telemetry`                         | array  | no       | The list of telemetry data from previous request (max_len: **10**)             |                                      |
| `telemetry.model_engine`            | string | no       | The model engine used for completions (max_len: **50**)                        | `vertex-ai`                          |
| `telemetry.model_name`              | string | no       | The model name used for completions (max_len: **50**)                          | `code-gecko`                         |
| `telemetry.lang`                    | string | no       | The language used for completions (max_len: **50**)                            | `python`                             |
| `telemetry.experiments`             | array  | no       | The list of experiments run from previous request                              |                                      |
| `telemetry.experiments.name`        | string | yes      | The experiment name                                                            | `exp_truncate_suffix`                |
| `telemetry.experiments.variant`     | int    | yes      | The experiment variant                                                         | `0`                                  |
| `telemetry.requests`                | int    | yes      | The number of previously requested completions                                 | `1`                                  |
| `telemetry.accepts`                 | int    | yes      | The number of previously accepted completions                                  | `1`                                  |
| `telemetry.errors`                  | int    | yes      | The number of previously failed completions                                    | `0`                                  |

```shell
curl --request POST \
  --url 'https://codesuggestions.gitlab.com/v2/code/completions' \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "prompt_version": 2,
    "project_path": "gitlab-org/gitlab-shell",
    "project_id": 33191677,
    "model_provider": "anthropic",
    "current_file": {
      "file_name": "test.py",
      "content_above_cursor": "def is_even(n: int) ->",
      "content_below_cursor": ""
    }
    "telemetry": [
      {
        "model_engine": "anthropic",
        "model_name": "claude-instant-1.2",
        "lang": "python",
        "experiments": [
          {
            "name": "exp_truncate_suffix",
            "variant": 0
          }
        ],
        "requests": 1,
        "accepts": 1,
        "errors": 0
      }
    ]
  }'
```

Example response:

```json
{
  "id": "id",
  "model": {
    "engine": "anthropic",
    "name": "claude-instant-1.2",
    "lang": "python"
  },
  "experiments": [
    {
      "name": "exp_truncate_suffix",
      "variant": 0
    }
  ]
  "object": "text_completion",
  "created": 1682031100,
  "choices": [
    {
      "text": " bool:\n    return n % 2 == 0\n\n\ndef is_odd",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```

#### Responses

- `200: OK` if the service returns some completions.
- `422: Unprocessable Entity` if the required attributes are missing.
- `401: Unauthorized` if the service fails to authenticate using the access token.

### Generations

Given a prompt, the service will return one suggestion. This endpoint supports
two versions of payloads.

- If `vertex-ai` model provider is selected, we uses `code-bison@latest`.
- If `anthropic` model provider is selected, we uses `claude-2.0`.

```plaintext
POST /v2/code/generations
POST /ai/v2/code/generations
```

#### V1 Prompt

This performs some pre-processing of the content before forwarding it to the
third-party model provider.

| Attribute                           | Type   | Required | Description                                                                    | Example                   |
| ----------------------------------- | ------ | -------- | ------------------------------------------------------------------------------ | ------------------------- |
| `prompt_version`                    | int    | yes      | The version of the prompt.                                                     | `1`                       |
| `project_path`                      | string | no       | The name of the project (max_len: **255**).                                    | `gitlab-orb/gitlab-shell` |
| `project_id`                        | int    | no       | The id of the project.                                                         | `33191677`                |
| `model_provider`                    | string | no       | The name of the model provider. Valid values are: `anthropic` and `vertex-ai`. | `vertex-ai`               |
| `current_file`                      | hash   | yes      | The data of the current file.                                                  |                           |
| `current_file.file_name`            | string | yes      | The name of the current file (max_len: **255**).                               | `README.md`               |
| `current_file.content_above_cursor` | string | yes      | The content above cursor (max_len: **100,000**).                               | `import numpy as np`      |
| `current_file.content_below_cursor` | string | yes      | The content below cursor (max_len: **100,000**).                               | `def __main__:\n`         |
| `telemetry`                         | array  | no       | The list of telemetry data from previous request (max_len: **10**).            |                           |
| `telemetry.model_engine`            | string | no       | The model engine used for completions (max_len: **50**).                       | `vertex-ai`               |
| `telemetry.model_name`              | string | no       | The model name used for completions (max_len: **50**).                         | `code-gecko`              |
| `telemetry.lang`                    | string | no       | The language used for completions (max_len: **50**).                           | `python`                  |
| `telemetry.experiments`             | array  | no       | The list of experiments run from previous request.                             |                           |
| `telemetry.experiments.name`        | string | yes      | The experiment name.                                                           | `exp_truncate_suffix`     |
| `telemetry.experiments.variant`     | int    | yes      | The experiment variant.                                                        | `0`                       |
| `telemetry.requests`                | int    | yes      | The number of previously requested completions.                                | `1`                       |
| `telemetry.accepts`                 | int    | yes      | The number of previously accepted completions.                                 | `1`                       |
| `telemetry.errors`                  | int    | yes      | The number of previously failed completions.                                   | `0`                       |

```shell
curl --request POST \
  --url 'https://codesuggestions.gitlab.com/v2/code/generations' \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "prompt_version": 1,
    "project_path": "gitlab-org/gitlab-shell",
    "project_id": 33191677,
    "current_file": {
      "file_name": "test.py",
      "content_above_cursor": "// Create a Python function to generate prime numbers\n",
      "content_below_cursor": ""
    }
    "telemetry": [
      {
        "model_engine": "vertex-ai",
        "model_name": "code-bison@latest",
        "lang": "python",
        "experiments": [
          {
            "name": "exp_truncate_suffix",
            "variant": 0
          }
        ],
        "requests": 1,
        "accepts": 1,
        "errors": 0
      }
    ]
  }'
```

Example response:

```json
{
  "id": "id",
  "model": {
    "engine": "vertex-ai",
    "name": "code-gecko",
    "lang": "python"
  },
  "experiments": [
    {
      "name": "exp_truncate_suffix",
      "variant": 0
    }
  ]
  "object": "text_completion",
  "created": 1682031100,
  "choices": [
    {
      "text": " bool:\n    return n % 2 == 0\n\n\ndef is_odd",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```

#### V2 Prompt

This accepts prebuilt `prompt` and forwards it directly to third-party provider.

| Attribute                           | Type   | Required | Description                                                                                            | Example                              |
| ----------------------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| `prompt_version`                    | int    | yes      | The version of the prompt                                                                              | `2`                                  |
| `project_path`                      | string | no       | The name of the project (max_len: **255**)                                                             | `gitlab-orb/gitlab-shell`            |
| `project_id`                        | int    | no       | The id of the project                                                                                  | `33191677`                           |
| `model_provider`                    | string | no       | The name of the model provider. Valid values are: `anthropic` and `vertex-ai`. Default to `vertex-ai`. | `anthropic`                          |
| `current_file`                      | hash   | yes      | The data of the current file                                                                           |                                      |
| `current_file.file_name`            | string | yes      | The name of the current file (max_len: **255**)                                                        | `README.md`                          |
| `current_file.content_above_cursor` | string | yes      | The content above cursor (max_len: **100,000**)                                                        | `import numpy as np`                 |
| `current_file.content_below_cursor` | string | yes      | The content below cursor (max_len: **100,000**)                                                        | `def __main__:\n`                    |
| `prompt`                            | string | yes      | The content of a prebuilt prompt                                                                       | `Human: You are a code assistant...` |
| `telemetry`                         | array  | no       | The list of telemetry data from previous request (max_len: **10**)                                     |                                      |
| `telemetry.model_engine`            | string | no       | The model engine used for completions (max_len: **50**)                                                | `vertex-ai`                          |
| `telemetry.model_name`              | string | no       | The model name used for completions (max_len: **50**)                                                  | `code-gecko`                         |
| `telemetry.lang`                    | string | no       | The language used for completions (max_len: **50**)                                                    | `python`                             |
| `telemetry.experiments`             | array  | no       | The list of experiments run from previous request                                                      |                                      |
| `telemetry.experiments.name`        | string | yes      | The experiment name                                                                                    | `exp_truncate_suffix`                |
| `telemetry.experiments.variant`     | int    | yes      | The experiment variant                                                                                 | `0`                                  |
| `telemetry.requests`                | int    | yes      | The number of previously requested completions                                                         | `1`                                  |
| `telemetry.accepts`                 | int    | yes      | The number of previously accepted completions                                                          | `1`                                  |
| `telemetry.errors`                  | int    | yes      | The number of previously failed completions                                                            | `0`                                  |

```shell
curl --request POST \
  --url 'https://codesuggestions.gitlab.com/v2/code/generations' \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "prompt_version": 2,
    "project_path": "gitlab-org/gitlab-shell",
    "project_id": 33191677,
    "model_provider": "anthropic",
    "current_file": {
      "file_name": "test.py",
      "content_above_cursor": "// Create a Python function to generate prime numbers\n",
      "content_below_cursor": ""
    },
    "prompt": "Human: You are a code assistant...",
    "telemetry": [
      {
        "model_engine": "anthropic",
        "model_name": "claude-2.0",
        "lang": "python",
        "experiments": [
          {
            "name": "exp_truncate_suffix",
            "variant": 0
          }
        ],
        "requests": 1,
        "accepts": 1,
        "errors": 0
      }
    ]
  }'
```

Example response:

```json
{
  "id": "id",
  "model": {
    "engine": "anthropic",
    "name": "claude-2.0",
    "lang": "python"
  },
  "experiments": [
    {
      "name": "exp_truncate_suffix",
      "variant": 0
    }
  ]
  "object": "text_completion",
  "created": 1682031100,
  "choices": [
    {
      "text": "def generate_primes(n):\n",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```

#### Responses

- `200: OK` if the service returns some completions.
- `422: Unprocessable Entity` if the required attributes are missing.
- `401: Unauthorized` if the service fails to authenticate using the access token.

## Chat

### Agent

Given a prompt, the service will return response received from an AI provider as is.

```plaintext
POST /v1/agent/chat
POST /ai/v1/agent/chat
```

| Attribute                                      | Type   | Required | Description                                                                                                                                     | Example                                               |
| ---------------------------------------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `prompt_components`                            | array  | yes      | The list of prompt components comliant with https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/index.html#protocol (max_len: **1**). |                                                       |
| `prompt_components.type`                       | string | yes      | The type of the prompt component (max_len: **255**).                                                                                            | `prompt`                                              |
| `prompt_components.payload`                    | hash   | yes      | The data of the current prompt component.                                                                                                       |                                                       |
| `prompt_components.payload.content`            | string | yes      | The complete AI prompt (max_len: **400 000**).                                                                                                  | `Human: Tell me a fun fact about ducks\n\nAssistant:` |
| `prompt_components.payload.provider`           | string | yes      | The AI provider for which the prompt is designed for. Valid value is: `anthropic`.                                                              | `anthropic`                                           |
| `prompt_components.payload.model`              | string | yes      | The AI model for which the prompt is designed for. Valid values are: `claude-2.0` and `claude-instant-1.2`.                                     | `claude-2.0`                                          |
| `prompt_components.prompt_components.metadata` | hash   | no       | The metadata of the prompt component. Only string - string key value pairs are accepted.                                                        |                                                       |
| `prompt_components.metadata.source`            | string | yes      | The source of the prompt compoment (max_len: **100**).                                                                                          | `GitLab EE`                                           |
| `prompt_components.metadata.version`           | string | yes      | The version of the source (max_len: **100**).                                                                                                   | `16.7.0`                                              |

```shell
curl --request POST \
  --url 'https://codesuggestions.gitlab.com/v1/chat/agent \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "prompt_components": [
    {
      "type":"prompt",
      "payload":{
        "content": "Human: Tell me a fun fact about ducts\n\n\Assistant:,
        "provider": "anthropic",
        "model": "claude-2.0"
      },
      "metadata": {
        "source": "GitLab EE",
        "version": "16.7.0"
      }
    }
 ]
}'
```

Example response:

```json
{
  "response": "Here's a fun fact about ducks:...",
  "metadata": {
    "provider": "anthropic",
    "model": "claude-2.0",
    "timestamp": 1702292323
  }
}
```

#### Responses

- `200: OK` if the service returns some completions.
- `422: Unprocessable Entity` if the required attributes are missing.
- `401: Unauthorized` if the service fails to authenticate using the access token.

## XRay

### Libraries

Given a prompt template, the service will return response received from an AI provider as is.

```plaintext
POST /v1/x-ray/libraries
POST /ai/v1/x-ray/libraries
```

| Attribute                            | Type   | Required | Description                                                                                                                                     | Example                               |
| ------------------------------------ | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `prompt_components`                  | array  | yes      | The list of prompt components comliant with https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/index.html#protocol (max_len: **1**). |                                       |
| `prompt_components.type`             | string | yes      | The type of the prompt component (max_len: **255**).                                                                                            | `x_ray_package_file_prompt`           |
| `prompt_components.payload`          | hash   | yes      | The data of the prompt component.                                                                                                               |                                       |
| `prompt_components.payload.prompt`   | string | yes      | The complete AI prompt.                                                                                                                         | `Human: Tell me fun fact about ducks` |
| `prompt_components.payload.provider` | string | yes      | The AI provider for which the prompt is designed for.                                                                                           | `anthropic`                           |
| `prompt_components.payload.model`    | string | yes      | The AI model for which the prompt is designed for.                                                                                              | `claude-2.0`                          |
| `prompt_components.metadata`         | hash   | no       | The metadata of the prompt component. Only string - string key value pairs are accepted (max_len: **10**).                                      |                                       |

````shell
curl --request POST \
  --url 'https://codesuggestions.gitlab.com/v1/x-ray/libraries' \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "prompt_components": [
    {
      "type":"x_ray_package_file_prompt",
      "payload":{
        "prompt": "Human: Parse following content of Gemfile. Respond using only valid JSON with list of libraries available to use and their short description\n\nGemfile content:\n\n```\n gem kaminari\n```\n\n Assistant: {{\n\"libraries\":[{{\"name\": \"",
        "provider": "anthropic",
        "model": "claude-2.0"
      },
      "metadata": {
        "scannerVersion": "0.0.1"
      }
    }
 ]
}'
````

Example response:

````json
{
  "response": "Here is the response in valid JSON format with the list of libraries parsed from the Gemfile content:\n\n```json\n[\n  \"kaminari\"\n]\n```"
}
````

#### Responses

- `200: OK` if the service returns some completions.
- `422: Unprocessable Entity` if the required attributes are missing.
- `401: Unauthorized` if the service fails to authenticate using the access token.
