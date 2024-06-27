from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import RootModel

from ai_gateway.agents import Agent, BaseAgentRegistry
from ai_gateway.api.feature_category import feature_category
from ai_gateway.async_dependency_resolver import get_container_application
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.gitlab_features import GitLabFeatureCategory, WrongUnitPrimitives


class AgentRequest(RootModel):
    root: dict[str, Any]


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

    # We don't use `isinstance` because we don't want to match subclasses
    if not type(agent) is Agent:  # pylint: disable=unidiomatic-typecheck
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Agent '{agent_id}' is not supported",
        )

    try:
        response = await agent.ainvoke(agent_request.root)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    return response.content
