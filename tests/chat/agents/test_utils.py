import pytest

from ai_gateway.agents.base import Agent
from ai_gateway.chat.agents.utils import convert_prompt_to_messages
from ai_gateway.models import Message, Role


@pytest.mark.parametrize(
    ("agent", "prompt_kwargs", "expected"),
    [
        (
            Agent(name="Test", model=None, prompt_templates={"user": "user prompt"}),
            {},
            [Message(role=Role.USER, content="user prompt")],
        ),
        (
            Agent(
                name="Test",
                model=None,
                prompt_templates={"user": "user prompt {{text}}"},
            ),
            {"text": "!"},
            [Message(role=Role.USER, content="user prompt !")],
        ),
        (
            Agent(
                name="Test",
                model=None,
                prompt_templates={"system": "system prompt", "user": "user prompt"},
            ),
            {},
            [
                Message(role=Role.SYSTEM, content="system prompt"),
                Message(role=Role.USER, content="user prompt"),
            ],
        ),
        (
            Agent(
                name="Test",
                model=None,
                prompt_templates={
                    "system": "system prompt",
                    "user": "user prompt",
                    "assistant": "assistant prompt",
                },
            ),
            {},
            [
                Message(role=Role.SYSTEM, content="system prompt"),
                Message(role=Role.USER, content="user prompt"),
                Message(role=Role.ASSISTANT, content="assistant prompt"),
            ],
        ),
        (
            Agent(
                name="Test",
                model=None,
                prompt_templates={
                    "system": "system prompt",
                    "user": "user prompt {{text}}",
                    "assistant": "assistant prompt",
                },
            ),
            {"text": "!"},
            [
                Message(role=Role.SYSTEM, content="system prompt"),
                Message(role=Role.USER, content="user prompt !"),
                Message(role=Role.ASSISTANT, content="assistant prompt"),
            ],
        ),
        (
            Agent(
                name="Test",
                model=None,
                prompt_templates={
                    "system": "system prompt {{text}}",
                    "user": "user prompt {{text}}",
                    "assistant": "assistant prompt {{text}}",
                },
            ),
            {"text": "!"},
            [
                Message(role=Role.SYSTEM, content="system prompt !"),
                Message(role=Role.USER, content="user prompt !"),
                Message(role=Role.ASSISTANT, content="assistant prompt !"),
            ],
        ),
    ],
)
def test_convert_prompt_to_messages(
    agent: Agent, prompt_kwargs: dict, expected: list[Message]
):
    actual = convert_prompt_to_messages(agent, **prompt_kwargs)
    assert actual == expected
