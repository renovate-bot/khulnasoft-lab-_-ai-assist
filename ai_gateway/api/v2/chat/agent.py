from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory, GitLabUnitPrimitive
from starlette.responses import StreamingResponse

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import X_GITLAB_VERSION_HEADER
from ai_gateway.api.v2.chat.typing import AgentRequest
from ai_gateway.async_dependency_resolver import (
    get_container_application,
    get_internal_event_client,
)
from ai_gateway.chat.agents import (
    AdditionalContext,
    AgentStep,
    AgentToolAction,
    ReActAgentInputs,
    TypeAgentEvent,
)
from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.internal_events import InternalEventsClient

__all__ = [
    "router",
]

from ai_gateway.structured_logging import get_request_logger

request_log = get_request_logger("chat")

router = APIRouter()


async def get_gl_agent_remote_executor():
    yield get_container_application().chat.gl_agent_remote_executor()


def authorize_additional_context(
    current_user: StarletteUser,
    additional_context: AdditionalContext,
    internal_event_client: InternalEventsClient,
):
    unit_primitive = GitLabUnitPrimitive[
        f"include_{additional_context.category}_context".upper()
    ]
    if current_user.can(unit_primitive):
        internal_event_client.track_event(
            f"request_{unit_primitive}",
            category=__name__,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unauthorized to access {unit_primitive}",
        )


def authorize_agent_request(
    current_user: StarletteUser,
    agent_request: AgentRequest,
    internal_event_client: InternalEventsClient,
):
    if current_user.can(GitLabUnitPrimitive.DUO_CHAT):
        internal_event_client.track_event(
            f"request_{GitLabUnitPrimitive.DUO_CHAT}",
            category=__name__,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access duo chat",
        )

    if agent_request.messages:
        for message in agent_request.messages:
            if message.additional_context:
                for ctx in message.additional_context:
                    authorize_additional_context(
                        current_user, ctx, internal_event_client
                    )


@router.post("/agent")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
async def chat(
    request: Request,
    agent_request: AgentRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    gl_agent_remote_executor: GLAgentRemoteExecutor[
        ReActAgentInputs, TypeAgentEvent
    ] = Depends(get_gl_agent_remote_executor),
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    authorize_agent_request(current_user, agent_request, internal_event_client)

    async def _stream_handler(stream_events: AsyncIterator[TypeAgentEvent]):
        async for event in stream_events:
            yield f"{event.dump_as_response()}\n"

    if agent_request.options:
        scratchpad = [
            AgentStep(
                action=AgentToolAction(
                    thought=step.thought.replace("\\_", "_"),
                    tool=step.tool.replace("\\_", "_", 1),
                    tool_input=step.tool_input,
                ),
                observation=step.observation,
            )
            for step in agent_request.options.agent_scratchpad.steps
        ]
    else:
        scratchpad = []

    inputs = ReActAgentInputs(
        messages=agent_request.messages,
        agent_scratchpad=scratchpad,
        model_metadata=agent_request.model_metadata,
        unavailable_resources=agent_request.unavailable_resources,
    )

    gl_version = request.headers.get(X_GITLAB_VERSION_HEADER, "")
    gl_agent_remote_executor.on_behalf(current_user, gl_version)

    request_log.info("Request to V2 Chat Agent", source=__name__, inputs=inputs)

    stream_events = gl_agent_remote_executor.stream(inputs=inputs)

    # When StreamingResponse is returned, clients get 200 even if there was an error during the process.
    # This is because the status code is returned before the actual process starts,
    # and there is no way to tell clients that the status code was changed after the streaming started.
    # Ref: https://github.com/encode/starlette/discussions/1739#discussioncomment-3094935.
    # If an exception is raised during the process, you will see `exception_message` field in the access log.
    return StreamingResponse(
        _stream_handler(stream_events), media_type="application/x-ndjson; charset=utf-8"
    )
