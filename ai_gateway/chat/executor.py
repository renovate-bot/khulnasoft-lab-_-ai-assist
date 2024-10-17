from typing import AsyncIterator, Generic, Protocol

import starlette_context
import structlog
from langchain_core.runnables import Runnable

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.chat.agents import (
    AgentToolAction,
    ReActAgent,
    TypeAgentEvent,
    TypeAgentInputs,
)
from ai_gateway.chat.base import BaseToolsRegistry
from ai_gateway.chat.tools import BaseTool
from ai_gateway.feature_flags import FeatureFlag, is_feature_enabled
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.prompts.typing import ModelMetadata

__all__ = [
    "TypeAgentFactory",
    "GLAgentRemoteExecutor",
]

_REACT_AGENT_AVAILABLE_TOOL_NAMES_CONTEXT_KEY = "duo_chat.agent_available_tools"

log = structlog.stdlib.get_logger("gl_agent_remote_executor")


class TypeAgentFactory(Protocol[TypeAgentInputs, TypeAgentEvent]):
    def __call__(
        self,
        *,
        model_metadata: ModelMetadata,
    ) -> Runnable[TypeAgentInputs, TypeAgentEvent]: ...


class GLAgentRemoteExecutor(Generic[TypeAgentInputs, TypeAgentEvent]):
    def __init__(
        self,
        *,
        agent_factory: TypeAgentFactory,
        tools_registry: BaseToolsRegistry,
        internal_event_client: InternalEventsClient,
    ):
        self.agent_factory = agent_factory
        self.tools_registry = tools_registry
        self.internal_event_client = internal_event_client
        self._tools: list[BaseTool] | None = None

    @property
    def tools(self) -> list[BaseTool]:
        if self._tools is None:
            self._tools = self.tools_registry.get_all()

        return self._tools

    @property
    def tools_by_name(self) -> list[BaseTool]:
        return {tool.name: tool for tool in self.tools}

    def on_behalf(self, user: StarletteUser, gl_version: str):
        # Access the user tools as soon as possible to raise an exception
        # (in case of invalid unit primitives) before starting the data stream.
        # Reason: https://github.com/tiangolo/fastapi/discussions/10138
        if not user.is_debug:
            self._tools = self.tools_registry.get_on_behalf(user, gl_version)

    async def stream(self, *, inputs: TypeAgentInputs) -> AsyncIterator[TypeAgentEvent]:
        inputs.tools = self.tools
        agent: ReActAgent = self.agent_factory(
            agent_inputs=inputs,
            model_metadata=inputs.model_metadata,
        )

        tools_by_name = self.tools_by_name

        starlette_context.context[_REACT_AGENT_AVAILABLE_TOOL_NAMES_CONTEXT_KEY] = list(
            tools_by_name.keys()
        )

        if is_feature_enabled(FeatureFlag.EXPANDED_AI_LOGGING):
            log.info("Processed inputs", source=__name__, inputs=inputs)

        async for event in agent.astream():
            yield event

            if isinstance(event, AgentToolAction) and event.tool in tools_by_name:
                tool = tools_by_name[event.tool]
                self.internal_event_client.track_event(
                    f"request_{tool.unit_primitive}",
                    category=__name__,
                )
