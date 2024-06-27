from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from autograph.agents import PlannerAgent, PlanSupervisorAgent
from autograph.agents.planning import _PLANNER_SYSTEM_PROMPT
from autograph.entities import (
    Cost,
    Plan,
    Task,
    TaskStatusEnum,
    TaskStatusInput,
    WorkflowState,
)


class TestPlannerAgent:
    def test_setup(self, agent_config, tools):
        chat_anthropic_mock = MagicMock(ChatAnthropic)
        chat_anthropic_class_mock = MagicMock(return_value=chat_anthropic_mock)
        chat_anthropic_mock.with_structured_output.return_value = "Set up model"

        with patch(
            "autograph.agents.planning.ChatAnthropic", chat_anthropic_class_mock
        ):
            planner_agent = PlannerAgent(
                config=agent_config, team=[agent_config], tools=tools
            )

            assert planner_agent._input_messages == [
                SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
                HumanMessage(
                    content="The team consist of: ['You are a helpful assistant.']"
                ),
                HumanMessage(
                    content="The team has access to following tools: ['Search the web for information', 'Perform mathematical calculations']"
                ),
            ]

            assert planner_agent._llm == "Set up model"
            chat_anthropic_class_mock.assert_called_once_with(
                model_name=agent_config.model, temperature=agent_config.temperature
            )
            chat_anthropic_mock.with_structured_output.assert_called_once_with(
                Plan.model_json_schema(), include_raw=True
            )

    @pytest.mark.asyncio
    async def test_run(self, agent_config, tools):
        model_response = {
            "parsed": {
                "steps": [
                    {
                        "description": "Do something",
                        "status": TaskStatusEnum.NOT_STARTED,
                    },
                    {
                        "description": "Do something else",
                        "status": TaskStatusEnum.NOT_STARTED,
                    },
                ]
            },
            "raw": AIMessage(
                content="Test plan",
                response_metadata={
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "total_tokens": 30,
                    }
                },
            ),
        }

        chat_anthropic_mock = AsyncMock(ChatAnthropic)
        chat_anthropic_mock.with_structured_output.return_value = chat_anthropic_mock
        chat_anthropic_mock.ainvoke.return_value = model_response

        with patch(
            "autograph.agents.planning.ChatAnthropic",
            MagicMock(return_value=chat_anthropic_mock),
        ):
            planner_agent = PlannerAgent(
                config=agent_config, team=[agent_config], tools=tools
            )
            state = WorkflowState(goal="Test goal")
            result = await planner_agent.run(state)

            chat_anthropic_mock.ainvoke.assert_called_once_with(
                [
                    SystemMessage(_PLANNER_SYSTEM_PROMPT),
                    HumanMessage(
                        content="The team consist of: ['You are a helpful assistant.']"
                    ),
                    HumanMessage(
                        content="The team has access to following tools: ['Search the web for information', 'Perform mathematical calculations']"
                    ),
                    HumanMessage(content="The goal is: Test goal"),
                ]
            )
            assert result["plan"] == [
                Task(description="Do something", status=TaskStatusEnum.NOT_STARTED),
                Task(
                    description="Do something else", status=TaskStatusEnum.NOT_STARTED
                ),
            ]
            assert result["costs"] == (
                agent_config.model,
                Cost(llm_calls=1, input_tokens=10, output_tokens=20),
            )


def _model_response(
    status=TaskStatusEnum.COMPLETED, input_tokens=10, output_tokens=20, total_tokens=30
) -> dict:
    return {
        "parsed": {"status": status},
        "raw": AIMessage(
            content=f"Task {status}",
            response_metadata={
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            },
        ),
    }


class TestPlanSupervisorAgent:
    def test_setup(self, agent_config, tools):
        chat_anthropic_mock = MagicMock(ChatAnthropic)
        chat_anthropic_class_mock = MagicMock(return_value=chat_anthropic_mock)
        chat_anthropic_mock.with_structured_output.return_value = chat_anthropic_mock
        prompt_template_mock = MagicMock(ChatPromptTemplate)

        with patch(
            "autograph.agents.planning.ChatAnthropic", chat_anthropic_class_mock
        ), patch(
            "autograph.agents.planning.ChatPromptTemplate", prompt_template_mock
        ), patch(
            "autograph.agents.planning._PLAN_SUPERVISOR_SYSTEM_PROMPT",
            "Plan Supervisor Agent system prompt",
        ):
            plan_supervisor_agent = PlanSupervisorAgent(config=agent_config)

            assert plan_supervisor_agent._llm is not None
            chat_anthropic_class_mock.assert_called_once_with(
                model_name=agent_config.model, temperature=agent_config.temperature
            )
            prompt_template_mock.from_messages.assert_called_once_with(
                [
                    ("system", "Plan Supervisor Agent system prompt"),
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
            chat_anthropic_mock.with_structured_output.assert_called_once_with(
                TaskStatusInput, include_raw=True
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "old_plan",
            "revised_tasks",
            "model_responses",
            "new_plan",
            "plan_review_summary",
        ),
        [
            # whole plan was completed, no task was cancelled of completed before revision
            (
                # old plan
                [
                    Task(description="Do something", status=TaskStatusEnum.NOT_STARTED),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.IN_PROGRESS,
                    ),
                ],
                # revised tasks
                [
                    Task(description="Do something", status=TaskStatusEnum.NOT_STARTED),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.IN_PROGRESS,
                    ),
                ],
                # model responses
                [
                    _model_response(status=TaskStatusEnum.COMPLETED),
                    _model_response(status=TaskStatusEnum.COMPLETED),
                ],
                # new plan
                [
                    Task(description="Do something", status=TaskStatusEnum.COMPLETED),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.COMPLETED,
                    ),
                ],
                # plan review summary
                HumanMessage(
                    content="All taksk were completed please call HandoverTool tool"
                ),
            ),
            # plan was partailly completed, not task was cancelled of completed before revision
            (
                # old plan
                [
                    Task(description="Do something", status=TaskStatusEnum.NOT_STARTED),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.IN_PROGRESS,
                    ),
                ],
                # revised tasks
                [
                    Task(description="Do something", status=TaskStatusEnum.NOT_STARTED),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.IN_PROGRESS,
                    ),
                ],
                # model responses
                [
                    _model_response(status=TaskStatusEnum.IN_PROGRESS),
                    _model_response(status=TaskStatusEnum.COMPLETED),
                ],
                # new plan
                [
                    Task(description="Do something", status=TaskStatusEnum.IN_PROGRESS),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.COMPLETED,
                    ),
                ],
                # plan review summary
                HumanMessage(
                    content=f"Next task to implement is: {Task(description='Do something', status=TaskStatusEnum.IN_PROGRESS)}"
                ),
            ),
            # whole plan was completed, two tasks were cancelled and completed before revision
            (
                # old plan
                [
                    Task(description="Do task 1", status=TaskStatusEnum.CANCELLED),
                    Task(description="Do task 2", status=TaskStatusEnum.COMPLETED),
                    Task(description="Do something", status=TaskStatusEnum.IN_PROGRESS),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.IN_PROGRESS,
                    ),
                ],
                # revised tasks
                [
                    Task(description="Do something", status=TaskStatusEnum.IN_PROGRESS),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.IN_PROGRESS,
                    ),
                ],
                # model responses
                [
                    _model_response(status=TaskStatusEnum.COMPLETED),
                    _model_response(status=TaskStatusEnum.COMPLETED),
                ],
                # new plan
                [
                    Task(description="Do task 1", status=TaskStatusEnum.CANCELLED),
                    Task(description="Do task 2", status=TaskStatusEnum.COMPLETED),
                    Task(description="Do something", status=TaskStatusEnum.COMPLETED),
                    Task(
                        description="Do something else",
                        status=TaskStatusEnum.COMPLETED,
                    ),
                ],
                # plan review summary
                HumanMessage(
                    content="All taksk were completed please call HandoverTool tool"
                ),
            ),
            # the model failed to respond in format
            (
                # old plan
                [
                    Task(description="Do something", status=TaskStatusEnum.NOT_STARTED),
                ],
                # revised tasks
                [
                    Task(description="Do something", status=TaskStatusEnum.NOT_STARTED),
                ],
                # model responses
                [
                    {k: v for k, v in _model_response().items() if k != "parsed"},
                ],
                # new plan
                [
                    Task(description="Do something", status=TaskStatusEnum.NOT_STARTED),
                ],
                # plan review summary
                HumanMessage(
                    content=f"Next task to implement is: {Task(description='Do something', status=TaskStatusEnum.NOT_STARTED)}"
                ),
            ),
        ],
    )
    async def test_run(
        self,
        agent_config,
        revised_tasks,
        model_responses,
        old_plan,
        new_plan,
        plan_review_summary,
    ):
        messages = [
            HumanMessage(content="Started working on task 1"),
            AIMessage(content="Task 1 is now in progress"),
            HumanMessage(content="Completed task 1"),
        ]
        state = WorkflowState(goal="Test goal", plan=old_plan, messages=messages)

        chat_anthropic_mock = AsyncMock(ChatAnthropic)
        chat_anthropic_mock.with_structured_output.return_value = chat_anthropic_mock
        chat_anthropic_mock.ainvoke.side_effect = model_responses
        prompt_template_mock = MagicMock(ChatPromptTemplate)

        with patch(
            "autograph.agents.planning.ChatAnthropic",
            MagicMock(return_value=chat_anthropic_mock),
        ), patch(
            "autograph.agents.planning.ChatPromptTemplate.from_messages",
            return_value=prompt_template_mock,
        ):
            formated_messages = [
                SystemMessage(content="Plan Supervisor Agent system prompt"),
                HumanMessage(content="The goal is: Test goal"),
            ]
            prompt_template_mock.format.return_value = formated_messages
            plan_supervisor_agent = PlanSupervisorAgent(config=agent_config)

            result = await plan_supervisor_agent.run(state)

            expected_format_calls = [
                call(
                    goal=state["goal"],
                    task=task,
                    status=task.status,
                    messages=messages,
                )
                for task in revised_tasks
            ]
            assert prompt_template_mock.format.call_count == len(expected_format_calls)
            prompt_template_mock.format.assert_has_calls(expected_format_calls)

            expected_model_calls = [call(formated_messages) for _ in revised_tasks]
            assert chat_anthropic_mock.ainvoke.call_count == len(expected_model_calls)
            chat_anthropic_mock.ainvoke.assert_has_calls(expected_model_calls)

            assert result["plan"] == new_plan
            expected_messages = [
                *messages,
                HumanMessage(
                    content=f"I've revised the plan, the current status is: {new_plan}"
                ),
                plan_review_summary,
            ]
            assert result["messages"] == expected_messages
