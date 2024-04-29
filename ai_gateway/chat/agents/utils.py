from typing import Any

from ai_gateway.chat.prompts import ChatPrompt
from ai_gateway.models import Message, Role

__all__ = [
    "convert_prompt_to_messages",
]


def convert_prompt_to_messages(prompt: ChatPrompt, **kwargs: Any) -> list[Message]:
    messages = []
    for role, content in [
        (Role.SYSTEM, prompt.system),
        (Role.USER, prompt.user),
        (Role.ASSISTANT, prompt.assistant),
    ]:
        if content is None:
            continue

        messages.append(Message(role=role, content=content.format(**kwargs)))

    return messages
