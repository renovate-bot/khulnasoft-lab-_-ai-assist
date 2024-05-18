from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, Request
from starlette.authentication import requires
from starlette.responses import StreamingResponse

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v2.chat.typing import AgentRequest, AgentStreamResponseEvent
from ai_gateway.async_dependency_resolver import get_container_application
from ai_gateway.chat.agents import AgentStep, AgentToolAction
from ai_gateway.chat.agents.react import (
    ReActAgentInputs,
    ReActAgentToolAction,
    TypeReActAgentAction,
)
from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.gitlab_features import GitLabFeatureCategory, GitLabUnitPrimitive

__all__ = [
    "router",
]


log = structlog.stdlib.get_logger("chat")

router = APIRouter()


async def get_gl_agent_remote_executor():
    yield get_container_application().chat.gl_agent_remote_executor()


@router.post("/agent")
@requires(GitLabUnitPrimitive.DUO_CHAT)
@feature_category(GitLabFeatureCategory.DUO_CHAT)
async def chat(
    request: Request,
    agent_request: AgentRequest,
    gl_agent_remote_executor: GLAgentRemoteExecutor[
        ReActAgentInputs, TypeReActAgentAction
    ] = Depends(get_gl_agent_remote_executor),
):
    async def _stream_handler(stream_actions: AsyncIterator[TypeReActAgentAction]):
        async for action in stream_actions:
            event_type = (
                "action"
                if isinstance(action, AgentToolAction)
                else "final_answer_delta"
            )

            event = AgentStreamResponseEvent(type=event_type, data=action)

            yield f"{event.model_dump_json()}\n"

    inputs = ReActAgentInputs(
        question=agent_request.prompt,
        chat_history=agent_request.options.chat_history,
        context=agent_request.options.context,
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
        for step in agent_request.options.agent_scratchpad.steps
    ]

    stream_actions = gl_agent_remote_executor.stream(
        inputs=inputs, scratchpad=scratchpad
    )

    return StreamingResponse(
        _stream_handler(stream_actions), media_type="text/event-stream"
    )
