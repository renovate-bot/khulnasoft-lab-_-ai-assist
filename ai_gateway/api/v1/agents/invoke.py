from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, RootModel
from starlette.responses import StreamingResponse

from ai_gateway.agents import Agent, BaseAgentRegistry
from ai_gateway.api.feature_category import feature_category
from ai_gateway.async_dependency_resolver import (
    get_container_application,
    get_internal_event_client,
)
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.gitlab_features import GitLabFeatureCategory, WrongUnitPrimitives
from ai_gateway.internal_events import InternalEventsClient


class AgentInputs(RootModel):
    root: dict[str, Any]


class AgentRequest(BaseModel):
    inputs: AgentInputs
    stream: Optional[bool] = False


router = APIRouter()


async def get_agent_registry():
    yield get_container_application().pkg_agents.agent_registry()


@router.post(
    "/{agent_id:path}",
    response_model=str,
    status_code=status.HTTP_200_OK,
)
@feature_category(GitLabFeatureCategory.AI_ABSTRACTION_LAYER)
async def agent(
    request: Request,
    agent_request: AgentRequest,
    agent_id: str,
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
    agent_registry: Annotated[BaseAgentRegistry, Depends(get_agent_registry)],
    internal_event_client: InternalEventsClient = Depends(get_internal_event_client),
):
    try:
        agent = agent_registry.get_on_behalf(current_user, agent_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found",
        )
    except WrongUnitPrimitives:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unauthorized to access '{agent_id}'",
        )

    for unit_primitive in agent.unit_primitives:
        internal_event_client.track_event(
            f"request_{unit_primitive}",
            category=__name__,
        )

    # We don't use `isinstance` because we don't want to match subclasses
    if not type(agent) is Agent:  # pylint: disable=unidiomatic-typecheck
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Agent '{agent_id}' is not supported",
        )

    try:
        if agent_request.stream:
            response = agent.astream(agent_request.inputs.root)

            async def _handle_stream():
                async for chunk in response:
                    yield chunk.content

            return StreamingResponse(_handle_stream(), media_type="text/event-stream")

        response = await agent.ainvoke(agent_request.inputs.root)
        return response.content
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
