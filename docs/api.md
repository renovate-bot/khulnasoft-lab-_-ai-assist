# API

An interactive API documentation site is available at <http://127.0.0.1:5001/docs>.
For more information, please see the official FastAPI [doc](https://fastapi.tiangolo.com/tutorial/first-steps/?h=interactive+api+docs#interactive-api-docs).

## Authentication

The Code Suggestions API supports authentication via Code Suggestions access tokens.

You can use an Code Suggestions access token to authenticate with the API by passing it in the
`Authorization` header and specifying the `X-Gitlab-Authentication-Type` header.

```shell
curl --header "Authorization: Bearer <access_token>" --header "X-Gitlab-Authentication-Type: oidc" \
  "http://localhost:5052/v2/code/completions"
```

## Code Suggestions

### V4

We have updated the endpoint name from `completions` to `suggestions` to avoid confusion with `code completions`. Currently, the v4 endpoint is functionally equivalent to the [v3](#v3) endpoint. We plan to modify it later in the issue [Convert stream to Server-Side Events format](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/372).

```plaintext
POST /v4/code/suggestions
```

### V3

The v3 endpoint is aligned to the [architectural blueprint](https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/index.html#example-feature-code-suggestions).

```plaintext
POST /v3/code/completions
```

#### Completion

| Attribute                                        | Type    | Required | Description                                                                                        | Example                  |
| ------------------------------------------------ | ------- | -------- | -------------------------------------------------------------------------------------------------- | ------------------------ |
| `prompt_components.type`                         | string  | yes      | Identifies the prompt_payload type (for completions use `code_editor_completion`)                  | `code_editor_completion` |
| `prompt_components.payload.choices_count`        | int     | no       | The number of code completion choices to return (max_len: **4**). Only applies for `vertex-ai`. Does not support streaming. | `2`                                   |
| `prompt_components.payload.file_name`            | string  | yes      | The name of the current file (max_len: **255**)                                                    | `README.md`              |
| `prompt_components.payload.content_above_cursor` | string  | yes      | The content above cursor (max_len: **100,000**)                                                    | `import numpy as np`     |
| `prompt_components.payload.content_below_cursor` | string  | yes      | The content below cursor (max_len: **100,000**)                                                    | `def __main__:\n`        |
| `prompt_components.payload.language_identifier`  | string  | no       | [Language identifier](https://code.visualstudio.com/docs/languages/identifiers) (max_len: **255**) | `python`                 |
| `prompt_components.payload.model_provider`       | string  | no       | The model engine that should be used for the completion                                            | `anthropic`              |
| `prompt_components.payload.stream`               | boolean | no       | Enables streaming response, if applicable (default: false)                                         | `true`                   |
| `prompt_components.metadata.source`              | string  | no       | Source of the completionrequest (max_len: **255**)                                                 | `GitLab EE`              |
| `prompt_components.metadata.version`             | string  | no       | Version of the source (max_len: **255**)                                                           | `16.3`                   |

```shell
curl --request POST \
  --url "http://localhost:5052/v3/code/completions" \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
      "prompt_components": [
        {
          "type": "code_editor_completion",
          "payload": {
            "file_name": "test",
            "content_above_cursor": "func hello_world(){\n\t",
            "content_below_cursor": "\n}",
            "model_provider": "vertex-ai",
            "language_identifier": "go",
            "choices_count": 3
          },
          "metadata": {
            "source": "Gitlab EE",
            "version": "16.3"
          }
        }
      ]
    }'
```

Example response:

```json
{
  "choices": [
    {
      "text": "fmt.Println(\"Hello World\")\n",
      "index": 0,
      "finish_reason": "length"
    }
  ],
  "metadata": {
    "model": {
      "engine": "vertex-ai",
      "name": "code-gecko@002",
      "lang": "go"
    },
    "timestamp": 1702389046
  }
}
```

#### Generation

| Attribute                                        | Type    | Required | Description                                                                                        | Example                              |
| ------------------------------------------------ |---------| -------- | -------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `prompt_components.type`                         | string  | yes      | Identifies the prompt_payload type (for generation use `code_editor_generation`)                   | `code_editor_generation`             |
| `prompt_components.payload.file_name`            | string  | yes      | The name of the current file (max_len: **255**)                                                    | `README.md`                          |
| `prompt_components.payload.content_above_cursor` | string  | yes      | The content above cursor (max_len: **100,000**)                                                    | `import numpy as np`                 |
| `prompt_components.payload.content_below_cursor` | string  | yes      | The content below cursor (max_len: **100,000**)                                                    | `def __main__:\n`                    |
| `prompt_components.payload.language_identifier`  | string  | no       | [Language identifier](https://code.visualstudio.com/docs/languages/identifiers) (max_len: **255**) | `python`                             |
| `prompt_components.payload.model_provider`       | string  | no       | The model engine that should be used for the completion                                            | `anthropic`                          |
| `prompt_components.payload.stream`               | boolean | no       | Enables streaming response, if applicable (default: false)                                         | `true`                               |
| `prompt_components.payload.prompt`               | string  | no       | An optional pre-built prompt to be passed directly to the model (max_len: **400,000**)             | `Human: You are a code assistant...` |
| `prompt_components.payload.prompt_enhancer`      | JSON    | no       | Parameters to enhance the prompt, eg, examples, contexts, libraries, instructions                                                |`{"examples_array": [], "trimmed_content_above_cursor": "", "trimmed_content_below_cursor": "", "related_files": [],  "related_snippets": [], "libraries": []}`                          |
| `prompt_components.metadata.source`              | string  | no       | Source of the completionrequest (max_len: **255**)                                                 | `GitLab EE`                          |
| `prompt_components.metadata.version`             | string  | no       | Version of the source (max_len: **255**)                                                           | `16.3`                               |

```shell
curl --request POST \
  --url "http://localhost:5052/v3/code/completions" \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
      "prompt_components": [
        {
          "type": "code_editor_generation",
          "payload": {
            "file_name": "test",
            "content_above_cursor": "func hello_world(){\n\t",
            "content_below_cursor": "\n}",
            "model_provider": "anthropic",
            "language_identifier": "go",
            "prompt": "Human: Write a golang function that prints hello world.",
            "prompt_enhancer": {
              "examples_array":[
                  {
                    "example":"// calculate the square root of a number",
                    "response":"<new_code>var primes []int\n  for _, num := range list {\n    isPrime := true\n    for i := 2; i <= num/2; i++ {\n      if num%i == 0 {\n        isPrime = false\n        break\n      }\n    }\n    if isPrime { primes = append(primes, num) }\n  }",
                    "trigger_type":"comment"
                  }
              ],
              "trimmed_content_above_cursor":"package client\n\n// write a function to find min abs value from an array\n",
              "trimmed_content_below_cursor":"",
              "related_files":[
                  "<file_content file_name=\"client/tgao.go\">\n// write </file_content>",
              ],
              "related_snippets":[
                  "<snippet_content name=\"code snippet1\">\n//def test</snippet_content>",
              ],
              "libraries":[
                  "bytes",
                  "context"
              ],
              "user_instruction":"// write a function to find min abs value from an array"
            }
          },
          "metadata": {
            "source": "Gitlab EE",
            "version": "16.3"
          }
        }
      ]
    }'
```

Example response:

```json
{
  "choices": [
    {
      "text": "\n\nHere is a golang function that prints \"Hello World\":\n\n [...] printing \"Hello World\" to the console.",
      "index": 0,
      "finish_reason": "length"
    }
  ],
  "metadata": {
    "model": {
      "engine": "anthropic",
      "name": "claude-2.1",
      "lang": "go"
    },
    "timestamp": 1702389469
  }
}
```

Example streaming response:

```plaintext


Here is a golang function that prints "Hello World":

```go
package main

import "fmt"

func main() {
  fmt.Println("Hello World") 
}

[...]

This will compile and execute the program, printing "Hello World" to the console.
```

#### Responses

- `200: OK` if the service returns some completions.
- `401: Unauthorized` if the service fails to authenticate using the access token.
- `422: Unprocessable Entity` if the required attributes are missing or the number of `prompt_component` objects is not adequet.

### V2

#### Completions

Given a prompt, the service will return one suggestion. This endpoint supports
two versions of payloads.

- If `vertex-ai` model provider is selected, we use `code-gecko@002`.
- If `anthropic` model provider is selected, we use `claude-2.1`.

```plaintext
POST /v2/completions
POST /v2/code/completions
```

##### V1 Prompt

This performs some pre-processing of the content before forwarding it to the
third-party model provider.

| Attribute                           | Type   | Required | Description                                                                                                      | Example                   |
| ----------------------------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `prompt_version`                    | int    | yes      | The version of the prompt.                                                                                       | `1`                       |
| `project_path`                      | string | no       | The name of the project (max_len: **255**).                                                                      | `gitlab-orb/gitlab-shell` |
| `project_id`                        | int    | no       | The ID of the project.                                                                                           | `33191677`                |
| `model_provider`                    | string | no       | The name of the model provider. Valid values are: `anthropic` and `vertex-ai`.                                   | `vertex-ai`               |
| `current_file`                      | hash   | yes      | The data of the current file.                                                                                    |                           |
| `current_file.file_name`            | string | yes      | The name of the current file (max_len: **255**).                                                                 | `README.md`               |
| `current_file.language_identifier`  | string | no       | The language identifier defined from editor (max_len: **255**). This overrides language detected from file name. | `python`                  |
| `current_file.content_above_cursor` | string | yes      | The content above cursor (max_len: **100,000**).                                                                 | `import numpy as np`      |
| `current_file.content_below_cursor` | string | yes      | The content below cursor (max_len: **100,000**).                                                                 | `def __main__:\n`         |
| `choices_count`                     | int    | no       | The number of code completion choices to return (max_len: **4**). Only applies for `vertex-ai`. Does not support streaming.        | `2`                                   |
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
  --url "http://localhost:5052/v2/code/completions" \
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
    },
    "choices_count": 2,
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
    },
    {
      "text": " bool:\n    return n % 2 == 0\n\n\ndef is_number_odd",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```

##### V2 Prompt

This accepts prebuilt `prompt` and forwards it directly to third-party provider. Prebuilt `prompt`s only supports `anthropic` model provider.

| Attribute                           | Type   | Required | Description                                                                    | Example                              |
| ----------------------------------- | ------ | -------- | ------------------------------------------------------------------------------ | ------------------------------------ |
| `prompt_version`                    | int    | yes      | The version of the prompt                                                      | `2`                                  |
| `project_path`                      | string | no       | The name of the project (max_len: **255**)                                     | `gitlab-orb/gitlab-shell`            |
| `project_id`                        | int    | no       | The ID of the project                                                          | `33191677`                           |
| `model_provider`                    | string | no       | The name of the model provider. Valid values are: `anthropic` and `vertex-ai`. | `anthropic`                          |
| `current_file`                      | hash   | yes      | The data of the current file                                                   |                                      |
| `current_file.file_name`            | string | yes      | The name of the current file (max_len: **255**)                                | `README.md`                          |
| `current_file.content_above_cursor` | string | yes      | The content above cursor (max_len: **100,000**)                                | `import numpy as np`                 |
| `current_file.content_below_cursor` | string | yes      | The content below cursor (max_len: **100,000**)                                | `def __main__:\n`                    |
| `prompt`                            | string | yes      | The content of a prebuilt prompt                                               | `Human: You are a code assistant...` |
| `choices_count`                     | int    | no       | The number of code completion choices to return (max_len: **4**). Only applies for `vertex-ai`. Does not support streaming. **Note:** The response may return a number of choices less than the `choices_count` as we drop suggestions with low scores.      | `2`                                   |
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
  --url "http://localhost:5052/v2/code/completions" \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "prompt_version": 2,
    "choices_count": 0,
    "project_path": "gitlab-org/gitlab-shell",
    "project_id": 33191677,
    "model_provider": "anthropic",
    "model_name": "claude-2.1",
    "current_file": {
      "file_name": "test.py",
      "content_above_cursor": "def is_even(n: int) ->",
      "content_below_cursor": ""
    }
    "telemetry": [
      {
        "model_engine": "anthropic",
        "model_name": "claude-2.1",
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
    "name": "claude-2.1",
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

##### Responses

- `200: OK` if the service returns some completions.
- `422: Unprocessable Entity` if the required attributes are missing.
- `401: Unauthorized` if the service fails to authenticate using the access token.

#### Generations

Given a prompt, the service will return one suggestion. This endpoint supports
two versions of payloads.

- If `vertex-ai` model provider is selected, we uses `code-bison@002`.
- If `anthropic` model provider is selected, we uses `claude-2.0`.

```plaintext
POST /v2/code/generations
```

##### V1 Prompt

This performs some pre-processing of the content before forwarding it to the
third-party model provider.

| Attribute                           | Type   | Required | Description                                                                                                                                                                            | Example                   |
| ----------------------------------- | ------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `prompt_version`                    | int    | yes      | The version of the prompt.                                                                                                                                                             | `1`                       |
| `project_path`                      | string | no       | The name of the project (max_len: **255**).                                                                                                                                            | `gitlab-orb/gitlab-shell` |
| `project_id`                        | int    | no       | The ID of the project.                                                                                                                                                                 | `33191677`                |
| `model_provider`                    | string | no       | The name of the model provider. Valid values are: `anthropic` and `vertex-ai`.                                                                                                         | `vertex-ai`               |
| `model_name`                        | string | no       | The name of the model name. Valid values are: `claude-2`, `claude-2.0`, `claude-2.1` if model_provider is `anthropic`.`code-bison`, `code-bison@002` if model_provider is `vertex-ai`. | `code-bison@002`          |
| `current_file`                      | hash   | yes      | The data of the current file.                                                                                                                                                          |                           |
| `current_file.file_name`            | string | yes      | The name of the current file (max_len: **255**).                                                                                                                                       | `README.md`               |
| `current_file.content_above_cursor` | string | yes      | The content above cursor (max_len: **100,000**).                                                                                                                                       | `import numpy as np`      |
| `current_file.content_below_cursor` | string | yes      | The content below cursor (max_len: **100,000**).                                                                                                                                       | `def __main__:\n`         |
| `telemetry`                         | array  | no       | The list of telemetry data from previous request (max_len: **10**).                                                                                                                    |                           |
| `telemetry.model_engine`            | string | no       | The model engine used for completions (max_len: **50**).                                                                                                                               | `vertex-ai`               |
| `telemetry.model_name`              | string | no       | The model name used for completions (max_len: **50**).                                                                                                                                 | `code-gecko`              |
| `telemetry.lang`                    | string | no       | The language used for completions (max_len: **50**).                                                                                                                                   | `python`                  |
| `telemetry.experiments`             | array  | no       | The list of experiments run from previous request.                                                                                                                                     |                           |
| `telemetry.experiments.name`        | string | yes      | The experiment name.                                                                                                                                                                   | `exp_truncate_suffix`     |
| `telemetry.experiments.variant`     | int    | yes      | The experiment variant.                                                                                                                                                                | `0`                       |
| `telemetry.requests`                | int    | yes      | The number of previously requested completions.                                                                                                                                        | `1`                       |
| `telemetry.accepts`                 | int    | yes      | The number of previously accepted completions.                                                                                                                                         | `1`                       |
| `telemetry.errors`                  | int    | yes      | The number of previously failed completions.                                                                                                                                           | `0`                       |

```shell
curl --request POST \
  --url "http://localhost:5052/v2/code/generations" \
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
        "model_name": "code-bison@002",
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
    "name": "code-bison@002",
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

##### V2 Prompt

This accepts prebuilt `prompt` and forwards it directly to third-party provider.

| Attribute                           | Type   | Required | Description                                                                                                                                                                            | Example                              |
| ----------------------------------- | ------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `prompt_version`                    | int    | yes      | The version of the prompt                                                                                                                                                              | `2`                                  |
| `project_path`                      | string | no       | The name of the project (max_len: **255**)                                                                                                                                             | `gitlab-orb/gitlab-shell`            |
| `project_id`                        | int    | no       | The ID of the project                                                                                                                                                                  | `33191677`                           |
| `model_provider`                    | string | no       | The name of the model provider. Valid values are: `anthropic` and `vertex-ai`. Default to `vertex-ai`.                                                                                 | `anthropic`                          |
| `model_name`                        | string | no       | The name of the model name. Valid values are: `claude-2`, `claude-2.0`, `claude-2.1` if model_provider is `anthropic`.`code-bison`, `code-bison@002` if model_provider is `vertex-ai`. | `claude-2.1`                         |
| `current_file`                      | hash   | yes      | The data of the current file                                                                                                                                                           |                                      |
| `current_file.file_name`            | string | yes      | The name of the current file (max_len: **255**)                                                                                                                                        | `README.md`                          |
| `current_file.content_above_cursor` | string | yes      | The content above cursor (max_len: **100,000**)                                                                                                                                        | `import numpy as np`                 |
| `current_file.content_below_cursor` | string | yes      | The content below cursor (max_len: **100,000**)                                                                                                                                        | `def __main__:\n`                    |
| `prompt`                            | string | yes      | The content of a prebuilt prompt                                                                                                                                                       | `Human: You are a code assistant...` |
| `telemetry`                         | array  | no       | The list of telemetry data from previous request (max_len: **10**)                                                                                                                     |                                      |
| `telemetry.model_engine`            | string | no       | The model engine used for completions (max_len: **50**)                                                                                                                                | `vertex-ai`                          |
| `telemetry.model_name`              | string | no       | The model name used for completions (max_len: **50**)                                                                                                                                  | `code-gecko`                         |
| `telemetry.lang`                    | string | no       | The language used for completions (max_len: **50**)                                                                                                                                    | `python`                             |
| `telemetry.experiments`             | array  | no       | The list of experiments run from previous request                                                                                                                                      |                                      |
| `telemetry.experiments.name`        | string | yes      | The experiment name                                                                                                                                                                    | `exp_truncate_suffix`                |
| `telemetry.experiments.variant`     | int    | yes      | The experiment variant                                                                                                                                                                 | `0`                                  |
| `telemetry.requests`                | int    | yes      | The number of previously requested completions                                                                                                                                         | `1`                                  |
| `telemetry.accepts`                 | int    | yes      | The number of previously accepted completions                                                                                                                                          | `1`                                  |
| `telemetry.errors`                  | int    | yes      | The number of previously failed completions                                                                                                                                            | `0`                                  |

```shell
curl --request POST \
  --url "http://localhost:5052/v2/code/generations" \
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

##### Responses

- `200: OK` if the service returns some completions.
- `422: Unprocessable Entity` if the required attributes are missing.
- `401: Unauthorized` if the service fails to authenticate using the access token.

## Chat

### Agent

Given a prompt, the service will return response received from an AI provider as is.

```plaintext
POST /v1/agent/chat
```

| Attribute                                      | Type   | Required | Description                                                                                                                                     | Example                                               |
| ---------------------------------------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `prompt_components`                            | array  | yes      | The list of prompt components compliant with <https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/index.html#protocol> (max_len: **1**). |                                                       |
| `prompt_components.type`                       | string | yes      | The type of the prompt component (max_len: **255**).                                                                                            | `prompt`                                              |
| `prompt_components.payload`                    | hash   | yes      | The data of the current prompt component.                                                                                                       |                                                       |
| `prompt_components.payload.content`            | [string, array]  | yes      | The complete AI prompt (max_len: **400 000**). See [Claude Message API](https://docs.anthropic.com/en/api/messages) for conversation roles payload.models in `claude-3` family. | `content: "hi how are you"`       |
| `prompt_components.payload.provider`           | string | yes      | The AI provider for which the prompt is designed for. Valid value is: `anthropic`.                                                              | `anthropic`                                           |
| `prompt_components.payload.model`              | string | yes      | The AI model for which the prompt is designed for. Valid values are: `claude-3-5-sonnet-20240620`,`claude-3-sonnet-20240229`,`claude-3-haiku-2024030`, `claude-3-opus-20240229`, `claude-2.1`.         | `claude-2.0`                            |
| `prompt_components.prompt_components.metadata` | hash   | no       | The metadata of the prompt component. Only string - string key value pairs are accepted.                                                        |                                                       |
| `prompt_components.metadata.source`            | string | yes      | The source of the prompt component (max_len: **100**).                                                                                          | `GitLab EE`                                           |
| `prompt_components.metadata.version`           | string | yes      | The version of the source (max_len: **100**).                                                                                                   | `16.7.0`                                              |

```shell
curl --request POST \
  --url "http://localhost:5052/v1/chat/agent" \
  --header 'Authorization: Bearer <access_token>' \
  --header 'X-Gitlab-Authentication-Type: oidc' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "prompt_components": [
    {
      "type":"prompt",
      "payload": {
        "content": [
          {
            "role": "user",
            "content": "Hi, how are you?"
          }
        ],
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229"
      },
      "metadata": {
        "source": "GitLab EE",
        "version": "17.2.0"
      }
    }
 ]
}'
```

Example response:

```json
{
  "response": "Hi there! As an AI language model, I don't have feelings or emotions, \
               but I'm operating properly and ready to assist you with any questions or tasks you may have. How can I help you today?.",
  "metadata": {
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
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
```

| Attribute                            | Type   | Required | Description                                                                                                                                     | Example                               |
| ------------------------------------ | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `prompt_components`                  | array  | yes      | The list of prompt components compliant with <https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/index.html#protocol> (max_len: **1**). |                                       |
| `prompt_components.type`             | string | yes      | The type of the prompt component (max_len: **255**).                                                                                            | `x_ray_package_file_prompt`           |
| `prompt_components.payload`          | hash   | yes      | The data of the prompt component.                                                                                                               |                                       |
| `prompt_components.payload.prompt`   | string | yes      | The complete AI prompt.                                                                                                                         | `Human: Tell me fun fact about ducks` |
| `prompt_components.payload.provider` | string | yes      | The AI provider for which the prompt is designed for.                                                                                           | `anthropic`                           |
| `prompt_components.payload.model`    | string | yes      | The AI model for which the prompt is designed for.                                                                                              | `claude-2.0`                          |
| `prompt_components.metadata`         | hash   | no       | The metadata of the prompt component. Only string - string key value pairs are accepted (max_len: **10**).                                      |                                       |

```shell
curl --request POST \
  --url "http://localhost:5052/v1/x-ray/libraries" \
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
        "model": "claude-2.1"
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
