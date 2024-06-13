from typing import AsyncIterator, Generic, Protocol, Sequence

from langchain_core.runnables import Runnable

from ai_gateway.agents.chat import TypeAgentAction, TypeAgentInputs
from ai_gateway.auth import GitLabUser
from ai_gateway.chat.base import BaseToolsRegistry
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
        tools_registry: BaseToolsRegistry,
    ):
        self.agent_factory = agent_factory
        self.tools_registry = tools_registry
        self._tools: list[BaseTool] | None = None

    @property
    def tools(self) -> list[BaseTool]:
        if self._tools is None:
            self._tools = self.tools_registry.get_all()

        return self._tools

    def on_behalf(self, user: GitLabUser):
        # Access the user tools as soon as possible to raise an exception
        # (in case of invalid unit primitives) before starting the data stream.
        # Reason: https://github.com/tiangolo/fastapi/discussions/10138
        if not user.is_debug:
            self._tools = self.tools_registry.get_on_behalf(user)

    async def invoke(self, *, inputs: TypeAgentInputs) -> TypeAgentAction:
        agent = self.agent_factory(tools=self.tools, inputs=inputs)

        return await agent.ainvoke(inputs)

    async def stream(
        self, *, inputs: TypeAgentInputs
    ) -> AsyncIterator[TypeAgentAction]:
        agent = self.agent_factory(tools=self.tools, inputs=inputs)

        async for action in agent.astream(inputs):
            yield action
