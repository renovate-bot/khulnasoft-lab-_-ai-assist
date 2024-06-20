from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from ai_gateway.agents import Agent, BaseAgentRegistry
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE, feature_categories
from ai_gateway.async_dependency_resolver import get_container_application
from ai_gateway.auth.user import GitLabUser, get_current_user
from ai_gateway.gitlab_features import FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS


class AgentRequest(BaseModel):
    name: str
    inputs: dict[str, Any]


router = APIRouter()


async def get_agent_registry():
    yield get_container_application().pkg_agents.agent_registry()


@router.post(
    "/invoke",
    response_model=str,
    status_code=status.HTTP_200_OK,
)
@feature_categories(FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
async def agent(
    request: Request,
    agent_request: AgentRequest,
    current_user: Annotated[GitLabUser, Depends(get_current_user)],
    agent_registry: Annotated[BaseAgentRegistry, Depends(get_agent_registry)],
):
    unit_primitive = request.headers.get(X_GITLAB_UNIT_PRIMITIVE)

    if not current_user.can(unit_primitive):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unauthorized to access {unit_primitive}",
        )

    try:
        agent = agent_registry.get(unit_primitive, agent_request.name)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_request.name}' not found for {unit_primitive}",
        )

    # We don't use `isinstance` because we don't want to match subclasses
    if not type(agent) is Agent:  # pylint: disable=unidiomatic-typecheck
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Agent '{agent_request.name}' is not supported",
        )

    try:
        response = await agent.ainvoke(agent_request.inputs)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    return response.content
