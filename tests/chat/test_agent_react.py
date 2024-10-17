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
    ReActPlainTextParser,
)
from ai_gateway.chat.agents.typing import (
    AdditionalContext,
    AgentStep,
    Context,
    CurrentFile,
    Message,
)
from ai_gateway.chat.tools.gitlab import IssueReader
from ai_gateway.feature_flags.context import current_feature_flag_context
from ai_gateway.models.base_chat import Role


@pytest.fixture
def prompt_class():
    yield ReActAgent


@pytest.fixture
def prompt_kwargs():
    yield {
        "agent_inputs": ReActAgentInputs(
            messages=[
                Message(role=Role.USER, content="Hi, how are you?"),
                Message(role=Role.ASSISTANT, content="I'm good!"),
            ]
        ),
    }


async def _assert_agent_invoked(
    prompt: ReActAgent,
    messages: list[Message],
    agent_scratchpad: list[AgentStep],
    expected_actions: list[AgentToolAction | AgentFinalAnswer | AgentUnknownAction],
    stream: bool,
):
    inputs = ReActAgentInputs(
        messages=messages,
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
        with capture_logs() as cap_logs, request_cycle_context({}):
            actual_actions = [action async for action in prompt.astream()]

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
        actual_events = []
        with pytest.raises(ValueError) as exc_info:
            async for event in prompt.astream():
                actual_events.append(event)

        assert actual_events == expected_events
        assert str(exc_info.value) == error_message
