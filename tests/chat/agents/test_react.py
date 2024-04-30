from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from ai_gateway.agents import LocalAgentRegistry
from ai_gateway.chat.agents import AgentStep
from ai_gateway.chat.agents.react import (
    ReActAgent,
    ReActAgentFinalAnswer,
    ReActAgentInputs,
    ReActAgentMessage,
    ReActAgentToolAction,
    ReActPlainTextParser,
    agent_scratchpad_plain_text_renderer,
    chat_history_plain_text_renderer,
)
from ai_gateway.chat.agents.utils import convert_prompt_to_messages
from ai_gateway.chat.tools.gitlab import GitLabToolkit
from ai_gateway.chat.typing import Context
from ai_gateway.models import ChatModelBase, Role, SafetyAttributes, TextGenModelOutput


@pytest.fixture
def agent_registry() -> LocalAgentRegistry:
    return LocalAgentRegistry()


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (ReActAgentInputs(question="", chat_history=["str1", "str2"]), "str1\nstr2"),
        (ReActAgentInputs(question="", chat_history="str1\nstr2"), "str1\nstr2"),
    ],
)
def test_chat_history_plain_text_renderer(inputs: ReActAgentInputs, expected: str):
    actual = chat_history_plain_text_renderer(inputs)
    assert actual == expected


@pytest.mark.parametrize(
    ("scratchpad", "expected"),
    [
        (
            [
                AgentStep(
                    action=ReActAgentToolAction(
                        tool="tool1", tool_input="tool_input1", thought="thought1"
                    ),
                    observation="observation1",
                ),
                AgentStep(
                    action=ReActAgentToolAction(
                        tool="tool2", tool_input="tool_input2", thought="thought2"
                    ),
                    observation="observation2",
                ),
                AgentStep(
                    action=ReActAgentFinalAnswer(
                        text="final_answer", thought="thought3"
                    ),
                    observation="observation3",
                ),
            ],
            (
                "Thought: thought1\n"
                "Action: tool1\n"
                "Action Input: tool_input1\n"
                "Observation: observation1\n"
                "Thought: thought2\n"
                "Action: tool2\n"
                "Action Input: tool_input2\n"
                "Observation: observation2"
            ),
        )
    ],
)
def test_agent_scratchpad_plain_text_renderer(
    scratchpad: list[AgentStep], expected: str
):
    actual = agent_scratchpad_plain_text_renderer(scratchpad)

    assert actual == expected


class TestReActPlainTextParser:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "thought1\nAction: tool1\nAction Input: tool_input1\n",
                ReActAgentToolAction(
                    thought="thought1",
                    log="Thought: thought1\nAction: tool1\nAction Input: tool_input1\n",
                    tool="tool1",
                    tool_input="tool_input1",
                ),
            ),
            (
                "thought1\nFinal Answer: final answer\n",
                ReActAgentFinalAnswer(
                    thought="thought1",
                    log="Thought: thought1\nFinal Answer: final answer\n",
                    text="final answer",
                ),
            ),
        ],
    )
    def test_agent_message(self, text: str, expected: ReActAgentMessage):
        parser = ReActPlainTextParser()
        actual = parser.parse(text)

        assert actual == expected

    @pytest.mark.parametrize("text", ["random_text"])
    def test_error(self, text: str):
        parser = ReActPlainTextParser()

        with pytest.raises(ValueError):
            parser.parse(text)


class TestReActAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "question",
            "chat_history",
            "agent_scratchpad",
            "context",
            "expected_action",
        ),
        [
            (
                "What's the title of this epic?",
                "",
                [],
                Context(type="epic", content="epic title and description"),
                ReActAgentToolAction(
                    thought="I'm thinking...",
                    tool="ci_issue_reader",
                    tool_input="random input",
                    log="Thought: I'm thinking...\nAction: ci_issue_reader\nAction Input: random input",
                ),
            ),
            (
                "What's the title of this issue?",
                ["User: what's the description of this issue", "AI: PoC ReAct"],
                [],
                Context(type="issue", content="issue title and description"),
                ReActAgentToolAction(
                    thought="I'm thinking...",
                    tool="ci_issue_reader",
                    tool_input="random input",
                    log="Thought: I'm thinking...\nAction: ci_issue_reader\nAction Input: random input",
                ),
            ),
            (
                "What's the title of this issue?",
                ["User: what's the description of this issue", "AI: PoC ReAct"],
                [],
                None,
                ReActAgentToolAction(
                    thought="I'm thinking...",
                    tool="ci_issue_reader",
                    tool_input="random input",
                    log="Thought: I'm thinking...\nAction: ci_issue_reader\nAction Input: random input",
                ),
            ),
            (
                "What's your name?",
                "User: what's the description of this issue\nAI: PoC ReAct",
                [
                    AgentStep(
                        action=ReActAgentToolAction(
                            thought="thought",
                            tool="ci_issue_reader",
                            tool_input="random input",
                        ),
                        observation="observation",
                    )
                ],
                None,
                ReActAgentFinalAnswer(
                    thought="I'm thinking...",
                    text="Paris",
                    log="Thought: I'm thinking...\nFinal Answer: Paris",
                ),
            ),
        ],
    )
    async def test_success(
        self,
        agent_registry: LocalAgentRegistry,
        question: str,
        chat_history: list[str] | str,
        agent_scratchpad: list[AgentStep],
        context: Context | None,
        expected_action: ReActAgentToolAction | ReActAgentFinalAnswer,
    ):
        def _model_generate(*args, **kwargs):
            text = expected_action.log[
                len("Thought: ") :
            ]  # our default Assistant prompt template already contains "Thought: "
            return TextGenModelOutput(
                text=text, score=0, safety_attributes=SafetyAttributes()
            )

        model = Mock(spec=ChatModelBase)
        base_agent = agent_registry.get("chat", "react")
        inputs = ReActAgentInputs(
            question=question, chat_history=chat_history, context=context
        )

        model.generate = AsyncMock(side_effect=_model_generate)

        tools = GitLabToolkit().get_tools()
        agent = ReActAgent(agent=base_agent, model=model, inputs=inputs, tools=tools)
        agent.agent_scratchpad.extend(agent_scratchpad)
        actual_action = await agent.invoke(inputs=inputs)

        chat_history_formatted = chat_history_plain_text_renderer(inputs)
        agent_scratchpad_formatted = agent_scratchpad_plain_text_renderer(
            agent_scratchpad
        )
        messages = convert_prompt_to_messages(
            base_agent,
            tools=tools,
            context_type=(inputs.context.type if inputs.context else None),
            question=question,
            chat_history=chat_history_formatted,
            agent_scratchpad=agent_scratchpad_formatted,
            context_content=context.content if context else "",
        )
        messages = {message.role: message for message in messages}

        model.generate.assert_called_once_with(
            list(messages.values()), stream=False, stop_sequences=["Observation:"]
        )

        assert chat_history_formatted in messages[Role.SYSTEM].content
        assert context.content in messages[Role.SYSTEM].content if context else True
        assert (
            "{context_content}" not in messages[Role.SYSTEM].content
            if not context
            else True
        )
        assert question in messages[Role.USER].content
        assert agent_scratchpad_formatted in messages[Role.ASSISTANT].content
        assert actual_action == expected_action
