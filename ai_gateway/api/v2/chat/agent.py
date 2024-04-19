import structlog
from fastapi import APIRouter, Depends, Request, status
from starlette.authentication import requires

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v2.chat.typing import AgentRequest, AgentResponse, ReActAgentAction
from ai_gateway.async_dependency_resolver import get_container_application
from ai_gateway.chat.agents import AgentStep
from ai_gateway.chat.agents.react import (
    ReActAgentInputs,
    ReActAgentToolAction,
    TypeReActAgentAction,
)
from ai_gateway.chat.executor import GLAgentRemoteExecutor

__all__ = [
    "router",
]


log = structlog.stdlib.get_logger("chat")

router = APIRouter()


async def get_gl_agent_remote_executor():
    yield get_container_application().chat.gl_agent_remote_executor()


@router.post("/agent", response_model=AgentResponse, status_code=status.HTTP_200_OK)
@requires("duo_chat")
@feature_category("duo_chat")
async def chat(
    request: Request,
    agent_request: AgentRequest,
    gl_agent_remote_executor: GLAgentRemoteExecutor[
        ReActAgentInputs, TypeReActAgentAction
    ] = Depends(get_gl_agent_remote_executor),
):
    inputs = ReActAgentInputs(
        question=agent_request.question,
        chat_history=agent_request.chat_history,
        context=agent_request.context,
    )

    scratchpad = [
        AgentStep(
            action=ReActAgentToolAction(
                thought=step.thought,
                tool=step.tool,
                tool_input=step.tool_input,
            ),
            observation=step.observation,
        )
        for step in agent_request.agent_scratchpad.steps
    ]

    action = await gl_agent_remote_executor.invoke(inputs=inputs, scratchpad=scratchpad)
    if isinstance(action, ReActAgentToolAction):
        step = ReActAgentAction.AgentToolAction(
            tool=action.tool, tool_input=action.tool_input
        )
    else:
        step = ReActAgentAction.AgentFinalAnswer(text=action.text)

    return AgentResponse(
        agent_action=ReActAgentAction(thought=action.thought, step=step),
        log=action.log,
    )
