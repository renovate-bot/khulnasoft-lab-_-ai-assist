import pytest

from ai_gateway.chat.agents.react import (
    AgentFinalAnswer,
    AgentToolAction,
    AgentUnknownAction,
    ReActAgent,
    ReActAgentInputs,
    ReActInputParser,
    ReActPlainTextParser,
    agent_scratchpad_plain_text_renderer,
    chat_history_plain_text_renderer,
)
from ai_gateway.chat.agents.typing import (
    AdditionalContext,
    AgentStep,
    Context,
    CurrentFile,
)
from ai_gateway.chat.tools.base import BaseTool


@pytest.fixture
def prompt_class():
    yield ReActAgent


async def _assert_agent_invoked(
    prompt: ReActAgent,
    question: str,
    agent_scratchpad: list[AgentStep],
    chat_history: list[str] | str,
    expected_actions: list[AgentToolAction | AgentFinalAnswer | AgentUnknownAction],
    stream: bool,
):
    inputs = ReActAgentInputs(
        question=question,
        chat_history=chat_history,
        agent_scratchpad=agent_scratchpad,
    )

    if stream:
        actual_actions = [action async for action in prompt.astream(inputs)]
    else:
        actual_actions = [await prompt.ainvoke(inputs)]

    assert actual_actions == expected_actions


@pytest.fixture
def prompt_template():
    yield {
        "system": "{{chat_history}}\n\nYou are a DevSecOps Assistant named 'GitLab Duo Chat' created by GitLab.",
        "user": "{{question}}",
        "assistant": "{{agent_scratchpad}}",
    }


@pytest.fixture
def tool_action(model_response: str):
    yield AgentToolAction(
        thought="I'm thinking...",
        tool="ci_issue_reader",
        tool_input="random input",
        log=model_response,
    )


@pytest.fixture
def final_answer(model_response: str):
    yield AgentFinalAnswer(
        thought="I'm thinking...",
        text="Paris",
        log=model_response,
    )


def test_react_input_parser():
    additional_context = AdditionalContext(
        id="hello.py",
        category="file",
        content="def hello()-> str: return 'hello'",
        metadata={"file_type": "python"},
    )
    context = Context(
        type="issue",
        content="This is an incredibly interesting issue",
    )
    current_file = CurrentFile(
        file_path="/path/to/file.py",
        data="def hello_world():\n    print('Hello, World!')",
        selected_code=False,
    )

    inputs = ReActAgentInputs(
        question="What is this file about?",
        chat_history=["str1", "str2"],
        agent_scratchpad=[],
        additional_context=[additional_context],
        context=context,
        current_file=current_file,
        unavailable_resources=["Merge Requests", "Pipelines"],
        tools=[BaseTool(name="test_tool", description="A test tool")],
    )

    parser = ReActInputParser()
    parsed_inputs = parser.invoke(inputs)

    assert parsed_inputs["question"] == "What is this file about?"
    assert parsed_inputs["chat_history"] == "str1\nstr2"
    assert parsed_inputs["agent_scratchpad"] == ""
    assert parsed_inputs["additional_context"] == [additional_context]
    assert parsed_inputs["context_type"] == "issue"
    assert parsed_inputs["context_content"] == "This is an incredibly interesting issue"
    assert parsed_inputs["current_file"] == current_file
    assert parsed_inputs["unavailable_resources"] == ["Merge Requests", "Pipelines"]
    assert len(parsed_inputs["tools"]) == 1 and isinstance(
        parsed_inputs["tools"][0], BaseTool
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
                    action=AgentToolAction(
                        tool="tool1", tool_input="tool_input1", thought="thought1"
                    ),
                    observation="observation1",
                ),
                AgentStep(
                    action=AgentToolAction(
                        tool="tool2", tool_input="tool_input2", thought="thought2"
                    ),
                    observation="observation2",
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
                AgentToolAction(
                    thought="thought1",
                    tool="tool1",
                    tool_input="tool_input1",
                ),
            ),
            (
                "thought1\nFinal Answer: final answer\n",
                AgentFinalAnswer(
                    text="final answer",
                ),
            ),
            (
                "Hi, I'm GitLab Duo Chat.",
                AgentUnknownAction(
                    text="Hi, I'm GitLab Duo Chat.",
                ),
            ),
        ],
    )
    def test_agent_message(self, text: str, expected: AgentToolAction):
        parser = ReActPlainTextParser()
        actual = parser.parse(text)

        assert actual == expected


class TestReActAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "question",
            "chat_history",
            "agent_scratchpad",
            "model_response",
            "expected_action_fixture",
        ),
        [
            (
                "What's the title of this epic?",
                "",
                [],
                "Thought: I'm thinking...\nAction: ci_issue_reader\nAction Input: random input",
                "tool_action",
            ),
            (
                "What's the title of this issue?",
                ["User: what's the description of this issue", "AI: PoC ReAct"],
                [],
                "Thought: I'm thinking...\nAction: ci_issue_reader\nAction Input: random input",
                "tool_action",
            ),
            (
                "What's the title of this issue?",
                ["User: what's the description of this issue", "AI: PoC ReAct"],
                [],
                "Thought: I'm thinking...\nAction: ci_issue_reader\nAction Input: random input",
                "tool_action",
            ),
            (
                "What's your name?",
                "User: what's the description of this issue\nAI: PoC ReAct",
                [
                    AgentStep(
                        action=AgentToolAction(
                            thought="thought",
                            tool="ci_issue_reader",
                            tool_input="random input",
                        ),
                        observation="observation",
                    )
                ],
                "Thought: I'm thinking...\nFinal Answer: Paris",
                "final_answer",
            ),
        ],
    )
    async def test_invoke(
        self,
        request,
        question: str,
        chat_history: list[str] | str,
        agent_scratchpad: list[AgentStep],
        model_response: str,
        expected_action_fixture: str,
        prompt: ReActAgent,
    ):
        expected_action = request.getfixturevalue(expected_action_fixture)

        await _assert_agent_invoked(
            prompt=prompt,
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
                    AgentToolAction(
                        thought="I'm thinking...",
                        tool="ci_issue_reader",
                        tool_input="random input",
                    ),
                ],
            ),
            (
                "What's your name?",
                "User: what's the description of this issue\nAI: PoC ReAct",
                [
                    AgentStep(
                        action=AgentToolAction(
                            thought="thought",
                            tool="ci_issue_reader",
                            tool_input="random input",
                        ),
                        observation="observation",
                    )
                ],
                "Thought: I'm thinking...\nFinal Answer: Bar",
                [
                    AgentFinalAnswer(
                        text="B",
                    ),
                    AgentFinalAnswer(text="a"),
                    AgentFinalAnswer(text="r"),
                ],
            ),
            (
                "Hi, how are you? Do not include Final Answer:, Thought: and Action: in response.",
                "",
                [],
                "I'm good. How about you?",
                [
                    AgentUnknownAction(
                        text="I'm good. How about you?",
                    ),
                ],
            ),
        ],
    )
    async def test_stream(
        self,
        question: str,
        chat_history: list[str] | str,
        agent_scratchpad: list[AgentStep],
        model_response: str,
        expected_actions: list[AgentToolAction | AgentFinalAnswer],
        prompt: ReActAgent,
    ):
        await _assert_agent_invoked(
            prompt=prompt,
            question=question,
            chat_history=chat_history,
            agent_scratchpad=agent_scratchpad,
            stream=True,
            expected_actions=expected_actions,
        )
