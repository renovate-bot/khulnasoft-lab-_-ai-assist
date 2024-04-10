import pytest

from ai_gateway.chat.agents.utils import convert_prompt_to_messages
from ai_gateway.chat.prompts import ChatPrompt
from ai_gateway.models import Message, Role


@pytest.mark.parametrize(
    ("prompt", "prompt_kwargs", "expected"),
    [
        (
            ChatPrompt(user="user prompt"),
            {},
            [Message(role=Role.USER, content="user prompt")],
        ),
        (
            ChatPrompt(user="user prompt {text}"),
            {"text": "!"},
            [Message(role=Role.USER, content="user prompt !")],
        ),
        (
            ChatPrompt(system="system prompt", user="user prompt"),
            {},
            [
                Message(role=Role.SYSTEM, content="system prompt"),
                Message(role=Role.USER, content="user prompt"),
            ],
        ),
        (
            ChatPrompt(
                system="system prompt", user="user prompt", assistant="assistant prompt"
            ),
            {},
            [
                Message(role=Role.SYSTEM, content="system prompt"),
                Message(role=Role.USER, content="user prompt"),
                Message(role=Role.ASSISTANT, content="assistant prompt"),
            ],
        ),
        (
            ChatPrompt(
                system="system prompt",
                user="user prompt {text}",
                assistant="assistant prompt",
            ),
            {"text": "!"},
            [
                Message(role=Role.SYSTEM, content="system prompt"),
                Message(role=Role.USER, content="user prompt !"),
                Message(role=Role.ASSISTANT, content="assistant prompt"),
            ],
        ),
        (
            ChatPrompt(
                system="system prompt {text}",
                user="user prompt {text}",
                assistant="assistant prompt {text}",
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
    prompt: ChatPrompt, prompt_kwargs: dict, expected: list[Message]
):
    actual = convert_prompt_to_messages(prompt, **prompt_kwargs)
    assert actual == expected
