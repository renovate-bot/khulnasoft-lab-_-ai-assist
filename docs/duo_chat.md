# GitLab Duo Chat

This page explains the functionalities existing in AI Gateway for GitLab Duo Chat features.

## Process Flow

The following diagram illustrates the minimal process flow that is required to answer an essential user input (e.g. `Hello`):

```mermaid
sequenceDiagram
    autonumber

    participant User as User
    box gitlab.com
    participant GW as Websocket
    participant GR as GitLabRails
    participant GS as GitLabSidekiq
    participant CS as ChatStorage(Redis)
    end
    box cloud.gitlab.com/ai
    participant AG as AI Gateway API
    participant AP as AI Gateway Prompt
    end

    User->>GW: Establish connection
    activate GW
    User->>+GR: Ask a question (GraphQL API)
    GR->>GS: Schedule a background job
    GR-->>-User: OK
    GS->>GS: Start ReAct agent
    activate GS
    GS->>+CS: Retrieve chat history
    CS-->>-GS: return history
    GS->>+AG: Request to v2/chat/agent
    AG->>+AP: Request to a LLM
    AP-->>-AG: Return response
    AG-->>-GS: return final answer
    GS-->>GW: Publish the event
    GW-->>User: Get streamed response
    deactivate GS
    deactivate GW
```

### Detailed process flow

The following diagram illustrates more complicated process flow that is required to answer a real-world user input (e.g. `Summarize this issue`):

```mermaid
sequenceDiagram
    autonumber

    participant User as User
    box gitlab.com
    participant GW as Websocket
    participant GR as GitLabRails
    participant GS as GitLabSidekiq
    participant CS as ChatStorage(Redis)
    participant GD as GitLabDatabase(PostgreSQL)
    end
    box cloud.gitlab.com/ai
    participant AG as AI Gateway API
    participant AP as AI Gateway Prompt
    end

    User->>GW: Establish connection
    activate GW
    User->>+GR: Ask a question (GraphQL API)
    GR->>GS: Schedule a background job
    GR-->>-User: OK
    GS->>GS: Start ReAct agent
    activate GS
    GS->>+CS: Retrieve chat history
    CS-->>-GS: return history
    loop Execute ReAct steps
    GS->>+AG: Request to v2/chat/agent
    alt if LLM needs more context
    AG->>+AP: Request to a LLM
    AP-->>-AG: Return response
    AG-->>GS: return a tool info
    GS->>GS: Execute the tool
    GS->>+AG: Request to /v1/prompts/chat/<tool-name> for extracting the Resource ID from user input
    AG->>+AP: Request to a LLM
    AP-->>-AG: Return response
    AG-->>GS: return a resource ID
    GS->>GD: Request resource info
    GD-->>GS: Return resource info
    GS->>GS: Update the observation param and execute the next step
    else if LLM can give final answer
    AG->>+AP: Request to a LLM
    AP-->>-AG: Return response
    AG-->>GS: return final answer
    GS-->>GW: Publish the event
    GW-->>User: Get streamed response
    end
    end
    deactivate GS
    deactivate GW
```

## AI Gateway APIs

The following endpoints are used by GitLab Duo Chat:

- `v2/chat/agent` ... ReAct execution endpoint. This endpoint constructs a prompt with the tool definitions and request to LLM through [AI Gateway Prompt](./aigw_prompt_registry.md).
  - [AI Gateway Prompt configuration files](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/main/ai_gateway/prompts/definitions/chat/react?ref_type=heads)
- `/v1/prompts/chat/*` ... Slash commands and tool execution endpoints. See [AI Gateway Prompt](./aigw_prompt_registry.md) for more information.
  - [AI Gateway Prompt configuration files](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/main/ai_gateway/prompts/definitions/chat?ref_type=heads)

Visit `http://0.0.0.0:5052/docs` for the latest API interface of the Duo Chat endpoints.
