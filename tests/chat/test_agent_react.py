import pytest
from langchain.chat_models.fake import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.chat.agents.react import (
    ReActAgent,
    ReActAgentFinalAnswer,
    ReActAgentInputs,
    ReActAgentMessage,
    ReActAgentToolAction,
    ReActPlainTextParser,
    TypeReActAgentAction,
    agent_scratchpad_plain_text_renderer,
    chat_history_plain_text_renderer,
)
from ai_gateway.chat.agents.typing import AgentStep


async def _assert_agent_invoked(
    model_response: str | None,
    prompt_template: ChatPromptTemplate,
    question: str,
    agent_scratchpad: list[AgentStep],
    chat_history: list[str] | str,
    expected_actions: list[ReActAgentToolAction | ReActAgentFinalAnswer],
    stream: bool,
):
    # our default Assistant prompt template already contains "Thought: "
    text = "" if model_response is None else model_response[len("Thought: ") :]
    model = FakeListChatModel(responses=[text])

    inputs = ReActAgentInputs(
        question=question,
        chat_history=chat_history,
        agent_scratchpad=agent_scratchpad,
    )

    agent = ReActAgent(name="test", unit_primitives=[], chain=prompt_template | model)

    if stream:
        actual_actions = [action async for action in agent.astream(inputs)]
    else:
        actual_actions = [await agent.ainvoke(inputs)]

    assert actual_actions == expected_actions


@pytest.fixture
def prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{chat_history}\n\nYou are a DevSecOps Assistant named 'GitLab Duo Chat' created by GitLab.",
            ),
            ("user", "{question}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )


@pytest.mark.parametrize(
    ("chat_history", "expected"),
    [
        (["str1", "str2"], "str1\nstr2"),
        ("str1\nstr2", "str1\nstr2"),
    ],
)
def test_chat_history_plain_text_renderer(chat_history: str | list[str], expected: str):
    actual = chat_history_plain_text_renderer(chat_history)
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
                    log="Thought: thought1\nAction: tool1\nAction Input: tool_input1",
                    tool="tool1",
                    tool_input="tool_input1",
                ),
            ),
            (
                "thought1\nFinal Answer: final answer\n",
                ReActAgentFinalAnswer(
                    thought="thought1",
                    log="Thought: thought1\nFinal Answer: final answer",
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
            "expected_action",
        ),
        [
            (
                "What's the title of this epic?",
                "",
                [],
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
                    AgentStep[TypeReActAgentAction](
                        action=ReActAgentToolAction(
                            thought="thought",
                            tool="ci_issue_reader",
                            tool_input="random input",
                        ),
                        observation="observation",
                    )
                ],
                ReActAgentFinalAnswer(
                    thought="I'm thinking...",
                    text="Paris",
                    log="Thought: I'm thinking...\nFinal Answer: Paris",
                ),
            ),
        ],
    )
    async def test_invoke(
        self,
        prompt_template: ChatPromptTemplate,
        question: str,
        chat_history: list[str] | str,
        agent_scratchpad: list[AgentStep],
        expected_action: TypeReActAgentAction,
    ):
        await _assert_agent_invoked(
            model_response=expected_action.log,
            prompt_template=prompt_template,
            question=question,
            chat_history=chat_history,
            agent_scratchpad=agent_scratchpad,
            stream=False,
            expected_actions=[expected_action],
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "question",
            "chat_history",
            "agent_scratchpad",
            "model_response",
            "expected_actions",
        ),
        [
            (
                "What's the title of this epic?",
                "",
                [],
                "Thought: I'm thinking...\nAction: ci_issue_reader\nAction Input: random input",
                [
                    ReActAgentToolAction(
                        thought="I'm thinking...",
                        log="Thought: I'm thinking...\nAction: ci_issue_reader\nAction Input: random input",
                        tool="ci_issue_reader",
                        tool_input="random input",
                    ),
                ],
            ),
            (
                "What's your name?",
                "User: what's the description of this issue\nAI: PoC ReAct",
                [
                    AgentStep[TypeReActAgentAction](
                        action=ReActAgentToolAction(
                            thought="thought",
                            tool="ci_issue_reader",
                            tool_input="random input",
                        ),
                        observation="observation",
                    )
                ],
                "Thought: I'm thinking...\nFinal Answer: Bar",
                [
                    ReActAgentFinalAnswer(
                        thought="I'm thinking...",
                        log="Thought: I'm thinking...\nFinal Answer: B",
                        text="B",
                    ),
                    ReActAgentFinalAnswer(thought="", log="a", text="a"),
                    ReActAgentFinalAnswer(thought="", log="r", text="r"),
                ],
            ),
        ],
    )
    async def test_stream(
        self,
        prompt_template: ChatPromptTemplate,
        question: str,
        chat_history: list[str] | str,
        agent_scratchpad: list[AgentStep],
        model_response: str,
        expected_actions: list[ReActAgentToolAction | ReActAgentFinalAnswer],
    ):
        await _assert_agent_invoked(
            model_response=model_response,
            prompt_template=prompt_template,
            question=question,
            chat_history=chat_history,
            agent_scratchpad=agent_scratchpad,
            stream=True,
            expected_actions=expected_actions,
        )
