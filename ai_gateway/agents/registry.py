from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Optional, Type

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.agents.base import Agent, BaseAgentRegistry

__all__ = ["LocalAgentRegistry"]


class Key(NamedTuple):
    use_case: str
    type: str


class LocalAgentRegistry(BaseAgentRegistry):
    def __init__(self, agent_definitions: dict[Key, tuple[Type[Agent], dict]]):
        self.agent_definitions = agent_definitions

    @lru_cache
    def _get_model(self, provider: str, name: str) -> BaseChatModel:
        match provider:
            case "anthropic":
                return ChatAnthropic(model=name)  # type: ignore[call-arg]
            case _:
                raise ValueError(f"Unknown provider: {provider}")

    def get(
        self, use_case: str, agent_type: str, options: Optional[dict[str, Any]] = None
    ) -> Any:
        klass, config = self.agent_definitions[Key(use_case=use_case, type=agent_type)]

        model = self._get_model(
            provider=config["provider"],
            name=config["model"],
        ).bind(stop=config["stop"])

        messages = klass.build_messages(config["prompt_template"], options or {})
        prompt = ChatPromptTemplate.from_messages(messages)

        return klass(
            name=config["name"],
            chain=prompt | model,
        )

    @classmethod
    def from_local_yaml(cls, data: dict[Key, Type[Agent]]) -> "LocalAgentRegistry":
        agent_definitions = {}
        for key, klass in data.items():
            path = Path(__file__).parent / key.use_case / f"{key.type}.yml"
            with open(path, "r") as fp:
                agent_definitions[key] = (klass, yaml.safe_load(fp))

        return cls(agent_definitions)
