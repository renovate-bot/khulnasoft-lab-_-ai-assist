from typing import AsyncIterator, Generic, Protocol, Sequence

from langchain_core.runnables import Runnable

from ai_gateway.agents.chat import TypeAgentAction, TypeAgentInputs
from ai_gateway.chat.tools import BaseTool

__all__ = [
    "TypeAgentFactory",
    "GLAgentRemoteExecutor",
]


class TypeAgentFactory(Protocol[TypeAgentInputs, TypeAgentAction]):
    def __call__(
        self,
        *,
        tools: Sequence[BaseTool],
        inputs: TypeAgentInputs,
    ) -> Runnable[TypeAgentInputs, TypeAgentAction]: ...


class GLAgentRemoteExecutor(Generic[TypeAgentInputs, TypeAgentAction]):
    def __init__(
        self,
        *,
        agent_factory: TypeAgentFactory,
        tools: Sequence[BaseTool],
    ):
        self.agent_factory = agent_factory
        self.tools = tools

    async def invoke(self, *, inputs: TypeAgentInputs) -> TypeAgentAction:
        agent = self.agent_factory(tools=self.tools, inputs=inputs)

        return await agent.ainvoke(inputs)

    async def stream(
        self,
        *,
        inputs: TypeAgentInputs,
    ) -> AsyncIterator[TypeAgentAction]:
        agent = self.agent_factory(tools=self.tools, inputs=inputs)

        async for action in agent.astream(inputs):
            yield action
