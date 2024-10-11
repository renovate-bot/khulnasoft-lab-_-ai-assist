import pytest
from starlette_context import context, request_cycle_context
from structlog.testing import capture_logs

from ai_gateway.chat.agents.react import (
    AgentError,
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
    Message,
)
from ai_gateway.chat.tools.base import BaseTool
from ai_gateway.chat.tools.gitlab import IssueReader
from ai_gateway.feature_flags.context import current_feature_flag_context
from ai_gateway.models.base_chat import Message as LegacyMessage
from ai_gateway.models.base_chat import Role


@pytest.fixture
def prompt_class():
    yield ReActAgent


@pytest.fixture
def prompt_kwargs():
    yield {
        "chat_history": [
            LegacyMessage(role=Role.USER, content="Hi, how are you?"),
            LegacyMessage(role=Role.ASSISTANT, content="I'm good!"),
        ],
        "agent_inputs": ReActAgentInputs(),
    }


async def _assert_legacy_agent_invoked(
    prompt: ReActAgent,
    question: str,
    agent_scratchpad: list[AgentStep],
    chat_history: list[LegacyMessage] | list[str] | str,
    expected_actions: list[AgentToolAction | AgentFinalAnswer | AgentUnknownAction],
    stream: bool,
):
    inputs = ReActAgentInputs(
        question=question,
        chat_history=chat_history,
        agent_scratchpad=agent_scratchpad,
    )

    with capture_logs() as cap_logs, request_cycle_context({}):
        if stream:
            actual_actions = [action async for action in prompt.astream(inputs)]

            if isinstance(expected_actions[0], AgentToolAction):
                assert (
                    context.get("duo_chat.agent_tool_action")
                    == expected_actions[0].tool
                )
        else:
            actual_actions = [await prompt.ainvoke(inputs)]

    assert actual_actions == expected_actions

    if stream:
        assert cap_logs[-1]["event"] == "Response streaming"


@pytest.fixture
def prompt_template():
    yield {
        "system": "{% include 'chat/react/system.jinja' %}",
        "user": "{% include 'chat/react/user.jinja' %}",
        "assistant": "{% include 'chat/react/assistant.jinja' %}",
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


@pytest.fixture(autouse=True)
def stub_feature_flags():
    current_feature_flag_context.set(["expanded_ai_logging"])
    yield


@pytest.mark.parametrize(
    ("chat_history", "expected_chat_history"),
    [
        (["str1", "str2"], "str1\nstr2"),
        (
            [
                LegacyMessage(role=Role.USER, content="Hi, how are you?"),
                LegacyMessage(role=Role.ASSISTANT, content="I'm good!"),
            ],
            None,
        ),
    ],
)
def test_react_input_parser(
    chat_history: list[LegacyMessage] | list[str] | str, expected_chat_history
):
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
        chat_history=chat_history,
        agent_scratchpad=[],
        additional_context=[additional_context],
        context=context,
        current_file=current_file,
        unavailable_resources=["Merge Requests", "Pipelines"],
        tools=[BaseTool(name="test_tool", description="A test tool")],
    )

    parser = ReActInputParser()

    with capture_logs() as cap_logs:
        parsed_inputs = parser.invoke(inputs)

    assert parsed_inputs["question"] == "What is this file about?"
    if expected_chat_history:
        assert parsed_inputs["chat_history"] == expected_chat_history
    else:
        assert "chat_history" not in parsed_inputs
    assert parsed_inputs["agent_scratchpad"] == ""
    assert parsed_inputs["additional_context"] == [additional_context]
    assert parsed_inputs["context"].type == "issue"
    assert parsed_inputs["context"].content == "This is an incredibly interesting issue"
    assert parsed_inputs["current_file"] == current_file
    assert parsed_inputs["unavailable_resources"] == ["Merge Requests", "Pipelines"]
    assert len(parsed_inputs["tools"]) == 1 and isinstance(
        parsed_inputs["tools"][0], BaseTool
    )
    assert cap_logs[0]["event"] == "ReActInputParser"


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
            "prompt_kwargs",
            "model_response",
            "expected_actions",
        ),
        [
            (
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(
                                role=Role.USER,
                                content="What's the title of this issue?",
                            ),
                        ],
                        agent_scratchpad=[],
                        tools=[IssueReader()],
                    )
                },
                "Thought: I'm thinking...\nAction: issue_reader\nAction Input: random input",
                [
                    AgentToolAction(
                        thought="I'm thinking...",
                        tool="issue_reader",
                        tool_input="random input",
                    ),
                ],
            ),
            (
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(role=Role.USER, content="How can I log output?"),
                            Message(role=Role.ASSISTANT, content="Use print function"),
                            Message(
                                role=Role.USER,
                                content="Can you explain the print function?",
                            ),
                        ],
                        agent_scratchpad=[],
                    )
                },
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(text="A"),
                ],
            ),
            (
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(
                                role=Role.USER,
                                content="what's the description of this issue",
                            ),
                            Message(role=Role.ASSISTANT, content="PoC ReAct"),
                            Message(role=Role.USER, content="What's your name?"),
                        ],
                        agent_scratchpad=[
                            AgentStep(
                                action=AgentToolAction(
                                    thought="thought",
                                    tool="ci_issue_reader",
                                    tool_input="random input",
                                ),
                                observation="observation",
                            )
                        ],
                    )
                },
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
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(
                                role=Role.USER,
                                content="Explain this issue",
                                context=Context(
                                    type="issue", content="this issue is about Duo Chat"
                                ),
                            ),
                        ],
                    )
                },
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(
                        text="A",
                    ),
                ],
            ),
            (
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(
                                role=Role.USER,
                                content="Explain this code",
                                current_file=CurrentFile(
                                    file_path="main.py",
                                    data="print",
                                    selected_code=True,
                                ),
                            ),
                        ],
                    )
                },
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(
                        text="A",
                    ),
                ],
            ),
            (
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(
                                role=Role.USER,
                                content="Explain this code",
                                additional_context=[
                                    AdditionalContext(
                                        id="id",
                                        category="file",
                                        content="print",
                                        metadata={"a": "b"},
                                    )
                                ],
                            ),
                        ],
                    )
                },
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(
                        text="A",
                    ),
                ],
            ),
            (
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(
                                role=Role.USER,
                                content="Hi, how are you? Do not include Final Answer:, Thought: and Action: in response.",
                            ),
                        ],
                        agent_scratchpad=[],
                    )
                },
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
        prompt_kwargs: dict,
        model_response: str,
        expected_actions: list[AgentToolAction | AgentFinalAnswer | AgentUnknownAction],
        prompt: ReActAgent,
    ):
        inputs = prompt_kwargs["agent_inputs"]

        with capture_logs() as cap_logs, request_cycle_context({}):
            actual_actions = [action async for action in prompt.astream(inputs)]

            if isinstance(expected_actions[0], AgentToolAction):
                assert (
                    context.get("duo_chat.agent_tool_action")
                    == expected_actions[0].tool
                )

        assert actual_actions == expected_actions
        assert cap_logs[-1]["event"] == "Response streaming"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "prompt_kwargs",
            "model_error",
            "error_message",
            "expected_events",
        ),
        [
            (
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(
                                role=Role.USER,
                                content="What's the title of this epic?",
                            ),
                        ],
                    )
                },
                ValueError("overloaded_error"),
                "overloaded_error",
                [
                    AgentError(message="overloaded_error", retryable=True),
                ],
            ),
            (
                {
                    "agent_inputs": ReActAgentInputs(
                        messages=[
                            Message(
                                role=Role.USER,
                                content="What's the title of this epic?",
                            ),
                        ],
                    )
                },
                ValueError("api_error"),
                "api_error",
                [
                    AgentError(message="api_error", retryable=False),
                ],
            ),
        ],
    )
    async def test_stream_error(
        self,
        prompt_kwargs: dict,
        model_error: Exception,
        error_message: str,
        expected_events: list[AgentError],
        prompt: ReActAgent,
    ):
        inputs = prompt_kwargs["agent_inputs"]

        actual_events = []
        with pytest.raises(ValueError) as exc_info:
            async for event in prompt.astream(inputs):
                actual_events.append(event)

        assert actual_events == expected_events
        assert str(exc_info.value) == error_message


class TestLegacyReActAgent:
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

        await _assert_legacy_agent_invoked(
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
                "Can you explain the print function?",
                [
                    LegacyMessage(role=Role.USER, content="How can I log output?"),
                    LegacyMessage(role=Role.ASSISTANT, content="Use print function"),
                ],
                [],
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(text="A"),
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
        chat_history: list[LegacyMessage] | list[str] | str,
        agent_scratchpad: list[AgentStep],
        model_response: str,
        expected_actions: list[AgentToolAction | AgentFinalAnswer],
        prompt: ReActAgent,
    ):
        await _assert_legacy_agent_invoked(
            prompt=prompt,
            question=question,
            chat_history=chat_history,
            agent_scratchpad=agent_scratchpad,
            stream=True,
            expected_actions=expected_actions,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "question",
            "model_error",
            "error_message",
            "expected_events",
        ),
        [
            (
                "What's the title of this epic?",
                ValueError("overloaded_error"),
                "overloaded_error",
                [
                    AgentError(message="overloaded_error", retryable=True),
                ],
            ),
            (
                "What's the title of this epic?",
                ValueError("api_error"),
                "api_error",
                [
                    AgentError(message="api_error", retryable=False),
                ],
            ),
        ],
    )
    async def test_stream_error(
        self,
        question: str,
        model_error: Exception,
        error_message: str,
        expected_events: list[AgentError],
        prompt: ReActAgent,
    ):
        inputs = ReActAgentInputs(
            question=question,
            chat_history=[],
            agent_scratchpad=[],
        )

        actual_events = []
        with pytest.raises(ValueError) as exc_info:
            async for event in prompt.astream(inputs):
                actual_events.append(event)

        assert actual_events == expected_events
        assert str(exc_info.value) == error_message
