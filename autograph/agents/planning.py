from asyncio import gather
from typing import List

from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, PrivateAttr

from autograph.entities import (
    AgentConfig,
    Cost,
    Plan,
    Task,
    TaskStatusEnum,
    TaskStatusInput,
    WorkflowState,
)

__all__ = ["PlannerAgent", "PlanSupervisorAgent"]


_PLANNER_SYSTEM_PROMPT = f"""
For the given goal, come up with a simple step by step plan that can be deliver by the team appointed by user.
Only include steps that can be delivered by the assigned team with the assigned tools,\
when some additional steps smees to be required assume that they will be done afterwards by the user.  \
This plan should involve individual tasks, that if executed correctly will yield the correct answer.\
Do not add any superfluous steps. \
The result of the final step should be the call to HandoverTool.

You must respond with entity that match following JSONSchema
```schema
{Plan.model_json_schema()}
```
"""


class PlannerAgent(BaseModel):
    """Agent that plans the workflow"""

    config: AgentConfig
    _llm: Runnable = PrivateAttr()
    _input_messages: List[BaseMessage] = PrivateAttr()

    def __init__(self, config: AgentConfig, team: List[AgentConfig], tools: List[Tool]):
        super().__init__(
            config=AgentConfig(
                goal=config.goal,
                name="Planner",
                model=config.model,
                temperature=config.temperature,
                system_prompt=_PLANNER_SYSTEM_PROMPT,
                tools=config.tools,
            )
        )
        llm = ChatAnthropic(
            model_name=config.model, temperature=config.temperature
        )  # type: ignore
        self._llm = llm.with_structured_output(
            Plan.model_json_schema(), include_raw=True
        )

        self._input_messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(
                content=f"The team consist of: {[agent_config.system_prompt for agent_config in team]}"
            ),
            HumanMessage(
                content=f"The team has access to following tools: {[tool.description for tool in tools]}"
            ),
        ]

    async def run(self, state: WorkflowState):
        resp = await self._llm.ainvoke(
            self._input_messages
            + [HumanMessage(content=f"The goal is: {state['goal']}")]
        )
        usage_data = resp["raw"].response_metadata["usage"]
        return {
            "plan": [Task(**task) for task in resp["parsed"]["steps"]],
            "costs": (
                self.config.model,
                Cost(
                    llm_calls=1,
                    input_tokens=usage_data["input_tokens"],
                    output_tokens=usage_data["output_tokens"],
                ),
            ),
        }


_PLAN_SUPERVISOR_SYSTEM_PROMPT = f"""
You are an expert project manager. Your task is to oversee the execution of a plan by an experienced team member.
The plan consists of a set of Tasks, each with its own status.
You will review messages documenting the work completed by the team member for each Task presented to you.
Assign the correct status to each Task from the following options: {", ".join([status.value for status in TaskStatusEnum])}.
Only update the Task status if you are certain about the new status that should be applied.
"""


class PlanSupervisorAgent(BaseModel):
    """Agent that supervises the plan"""

    config: AgentConfig
    _llm: Runnable = PrivateAttr()
    _prompt_template: ChatPromptTemplate = PrivateAttr()

    def __init__(self, config: AgentConfig):
        super().__init__(
            config=AgentConfig(
                goal=config.goal,
                name="Plan supervisor",
                model=config.model,
                temperature=config.temperature,
                system_prompt=_PLAN_SUPERVISOR_SYSTEM_PROMPT,
                tools=config.tools,
            )
        )

        llm = ChatAnthropic(model_name=config.model, temperature=config.temperature)  # type: ignore

        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.config.system_prompt),
                ("human", "The goal is: {goal}"),
                (
                    "human",
                    "The current task to review is: {task}, the task previous state is {status}",
                ),
                (
                    "human",
                    "Here is conversation documentig all progress made so far: {messages}",
                ),
                ("human", "Please assign correct status to the task: {task}"),
            ]
        )
        self._llm = llm.with_structured_output(TaskStatusInput, include_raw=True)  # type: ignore[arg-type]

    async def run(self, state: WorkflowState):
        revised_plan = await gather(
            *[
                self._revise_task(task, state["goal"], state["messages"])
                for task in state["plan"]
            ]
        )

        open_tasks = [
            task
            for task in revised_plan
            if task.status not in (TaskStatusEnum.CANCELLED, TaskStatusEnum.COMPLETED)
        ]
        messages = list(state["messages"])
        messages.append(
            HumanMessage(
                content=f"I've revised the plan, the current status is: {revised_plan}"
            )
        )

        if len(open_tasks) > 0:
            messages.append(
                HumanMessage(content=f"Next task to implement is: {open_tasks[0]}")
            )
        else:
            messages.append(
                HumanMessage(
                    content="All taksk were completed please call HandoverTool tool"
                )
            )

        return {"plan": revised_plan, "messages": messages}

    async def _revise_task(
        self, task: Task, goal: str, messages: List[BaseMessage]
    ) -> Task:
        if task.status in (TaskStatusEnum.CANCELLED, TaskStatusEnum.COMPLETED):
            return task

        input_messages = self._prompt_template.format(
            goal=goal,
            task=task,
            status=task.status,
            messages=messages,
        )

        output = await self._llm.ainvoke(input_messages)

        new_status = output.get("parsed")

        if not new_status:
            # LLM output failed to parse into desired output
            return task

        return Task(description=task.description, status=new_status["status"])
