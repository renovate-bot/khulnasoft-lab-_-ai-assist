from typing import Generic, Protocol, Sequence

from ai_gateway.chat.agents import (
    AgentStep,
    BaseSingleActionAgent,
    TypeAgentAction,
    TypeAgentInputs,
)
from ai_gateway.chat.tools import BaseTool

__all__ = [
    "TypeAgentFactory",
    "GLAgentRemoteExecutor",
]


class TypeAgentFactory(Protocol[TypeAgentInputs]):
    def __call__(
        self,
        *,
        tools: Sequence[BaseTool],
        agent_inputs: TypeAgentInputs,
    ) -> BaseSingleActionAgent: ...


class GLAgentRemoteExecutor(Generic[TypeAgentInputs, TypeAgentAction]):
    def __init__(
        self,
        *,
        agent_factory: TypeAgentFactory,
        tools: Sequence[BaseTool],
    ):
        self.agent_factory = agent_factory
        self.tools = tools

    async def invoke(
        self,
        *,
        inputs: TypeAgentInputs,
        scratchpad: Sequence[AgentStep[TypeAgentAction]],
    ) -> TypeAgentAction:
        agent = self.agent_factory(tools=self.tools, agent_inputs=inputs)
        agent.agent_scratchpad.extend(scratchpad)

        action = await agent.invoke(inputs=inputs)

        return action
