from typing import Any

from ai_gateway.agents import Agent
from ai_gateway.models import Message, Role

__all__ = [
    "convert_prompt_to_messages",
]


def convert_prompt_to_messages(agent: Agent, **kwargs: Any) -> list[Message]:
    messages = []
    for role in Role:
        content = agent.prompt(role, **kwargs)
        if content is None:
            continue

        messages.append(Message(role=role, content=content))

    return messages
