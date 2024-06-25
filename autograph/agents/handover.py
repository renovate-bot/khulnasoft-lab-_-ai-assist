from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, PrivateAttr

from autograph.entities import AgentConfig, WorkflowState

__all__ = ["HandoverTool", "HandoverAgent"]


class HandoverTool(BaseModel):
    description: str = """A final response to the user"""
    summary: str = Field(
        description="The summary of the work done based on the past conversation between human, agent and tools executions"
    )


_DEFAULT_SYSTEM_PROMPT = """
    You are an expert manager. Your task is to ensure a smooth work handover between multiple team members.
    To achieve this, review past conversations between team members and summarize the progress that has been made towards the goal.
    Your summary should include the following information:
    1. What has been delivered so far.
    2. What is still missing or needs to be completed.
    3. Any problems or challenges that have been encountered.

    Your summary should be clear, concise, and provide a comprehensive overview of the project's current status to facilitate a seamless transition.
    """


class HandoverAgent(BaseModel):
    """Agent that summarizes the workflow"""

    config: AgentConfig
    _llm: Runnable = PrivateAttr()
    _prompt_template: ChatPromptTemplate = PrivateAttr()

    def __init__(self, config: AgentConfig):
        config_override = AgentConfig(
            goal=config.goal,
            name="Handover Agent",
            model=config.model,
            temperature=config.temperature,
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            tools=config.tools,
        )
        super().__init__(config=config_override)
        self._llm = ChatAnthropic(model_name=config.model, temperature=config.temperature)  # type: ignore
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", _DEFAULT_SYSTEM_PROMPT),
                ("human", "The goal is: {goal}"),
                ("human", "The conversation to summarize: {messages}"),
            ]
        )

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        input_messages = self._prompt_template.format(
            goal=state["goal"], messages=state["messages"]
        )
        response = await self._llm.ainvoke(input_messages)
        return {"previous_step_summary": response.content, "plan": [], "messages": []}
