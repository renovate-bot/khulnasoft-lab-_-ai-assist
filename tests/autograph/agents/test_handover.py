from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from autograph.agents import HandoverAgent
from autograph.entities import WorkflowState


class TestHandoverAgent:
    def test_setup(self, agent_config):
        chat_anthropic_mock = MagicMock(ChatAnthropic)
        chat_anthropic_class_mock = MagicMock(return_value=chat_anthropic_mock)
        prompt_template_mock = MagicMock(ChatPromptTemplate)

        with patch(
            "autograph.agents.handover.ChatAnthropic", chat_anthropic_class_mock
        ), patch(
            "autograph.agents.handover.ChatPromptTemplate", prompt_template_mock
        ), patch(
            "autograph.agents.handover._DEFAULT_SYSTEM_PROMPT",
            "Handover Agent system prompt",
        ):
            prompt_template_mock.from_messages.return_value = "prompt template"

            handover_agent = HandoverAgent(config=agent_config)

            assert handover_agent._llm == chat_anthropic_mock
            assert handover_agent._prompt_template == "prompt template"

            chat_anthropic_class_mock.assert_called_once_with(
                model_name=agent_config.model, temperature=agent_config.temperature
            )
            prompt_template_mock.from_messages.assert_called_once_with(
                [
                    ("system", "Handover Agent system prompt"),
                    ("human", "The goal is: {goal}"),
                    ("human", "The conversation to summarize: {messages}"),
                ]
            )

    @pytest.mark.asyncio
    async def test_run(self, agent_config):
        model_response = AIMessage(
            content="Handover summary",
            response_metadata={
                "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150}
            },
        )

        chat_anthropic_mock = AsyncMock(ChatAnthropic)
        chat_anthropic_mock.ainvoke.return_value = model_response
        prompt_template_mock = MagicMock(ChatPromptTemplate)
        with patch(
            "autograph.agents.handover.ChatAnthropic",
            MagicMock(return_value=chat_anthropic_mock),
        ), patch(
            "autograph.agents.handover.ChatPromptTemplate.from_messages",
            return_value=prompt_template_mock,
        ), patch(
            "autograph.agents.handover._DEFAULT_SYSTEM_PROMPT",
            "Handover Agent system prompt",
        ):
            input_messages = [
                SystemMessage("Handover Agent system prompt"),
                HumanMessage(content="The goal is: Test goal"),
                HumanMessage(
                    content="The conversation to summarize: ['Message 1', 'Message 2', 'Message 3']"
                ),
            ]
            prompt_template_mock.format.return_value = input_messages
            handover_agent = HandoverAgent(config=agent_config)
            state = WorkflowState(
                goal="Test goal", messages=["Message 1", "Message 2", "Message 3"]
            )

            result = await handover_agent.run(state)

            prompt_template_mock.format.assert_called_once_with(
                goal="Test goal", messages=["Message 1", "Message 2", "Message 3"]
            )
            chat_anthropic_mock.ainvoke.assert_called_once_with(input_messages)
            assert result == {
                "previous_step_summary": "Handover summary",
                "plan": [],
                "messages": [],
            }
