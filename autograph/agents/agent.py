import datetime
from typing import Any, Dict, List, Union

from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, PrivateAttr

from autograph.entities import AgentConfig, Cost, WorkflowState

__all__ = [
    "Agent",
]


class Agent(BaseModel):
    config: AgentConfig
    _llm: Union[Runnable, None] = PrivateAttr(default=None)

    def setup(self, tools: List[Tool]):
        llm = ChatAnthropic(
            model_name=self.config.model, temperature=self.config.temperature
        )  # type: ignore
        self._llm = llm.bind_tools(tools)

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        if not self._llm:
            raise ValueError("Agent not setup. Please call setup() first.")

        messages = state["messages"]
        if len(messages) <= 0:
            messages.append(SystemMessage(content=self.config.system_prompt))
            messages.append(
                HumanMessage(
                    content=f"Overall Goal: {state['goal']}\nYour Goal: {self.config.goal}"
                )
            )

        result = await self._llm.ainvoke(messages)
        actions = [
            {
                "actor": self.config.name,
                "contents": result.content,
                "time": str(datetime.datetime.now(datetime.timezone.utc)),
            }
        ]
        usage_data = result.response_metadata["usage"]

        return {
            "messages": messages + [result],
            "actions": actions,
            "costs": (
                self.config.model,
                Cost(
                    llm_calls=1,
                    input_tokens=usage_data["input_tokens"],
                    output_tokens=usage_data["output_tokens"],
                ),
            ),
        }
