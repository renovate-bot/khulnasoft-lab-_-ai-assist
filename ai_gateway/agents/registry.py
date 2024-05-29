from functools import lru_cache
from typing import Any, NamedTuple, Optional, Type

import yaml
from jinja2 import BaseLoader, Environment
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.agents.base import Agent, BaseAgentRegistry

__all__ = ["LocalAgentRegistry"]


class Key(NamedTuple):
    use_case: str
    type: str


class PromptTemplate(NamedTuple):
    user: str
    system: Optional[str] = None
    assistant: Optional[str] = None


class LocalAgentRegistry(BaseAgentRegistry):
    def __init__(self, agent_definitions: dict[Key, tuple[Type[Agent], dict]]):
        self.agent_definitions = agent_definitions
        self.jinja_env = Environment(loader=BaseLoader())

    @lru_cache
    def _get_model(self, provider: str, name: str) -> BaseChatModel:
        match provider:
            case "anthropic":
                return ChatAnthropic(model=name)  # type: ignore[call-arg]
            case _:
                raise ValueError(f"Unknown provider: {provider}")

    @lru_cache
    def _build_chat_prompt_templates(
        self, template: PromptTemplate, **kwargs: Optional[Any]
    ) -> ChatPromptTemplate:
        def _format_str(content: str, **kwargs: Optional[Any]) -> str:
            return self.jinja_env.from_string(content).render(**kwargs)

        messages = []
        for klass, content in [
            (SystemMessage, template.system),
            (HumanMessage, template.user),
            (AIMessage, template.assistant),
        ]:
            if content:
                messages.append(klass(content=_format_str(content, **kwargs)))

        return ChatPromptTemplate.from_messages(messages)

    def get(self, use_case: str, agent_type: str, **kwargs: Optional[Any]) -> Any:
        klass, config = self.agent_definitions[Key(use_case=use_case, type=agent_type)]

        model = self._get_model(
            provider=config["provider"],
            name=config["model"],
        ).bind(stop=config["stop"])

        # TODO: We build only chat prompt templates now
        prompt_template = self._build_chat_prompt_templates(
            PromptTemplate(**config["prompt_templates"]), **kwargs
        )

        return klass(
            name=config["name"],
            chain=prompt_template | model,
        )

    @classmethod
    def from_local_yaml(
        cls, data: dict[Key, tuple[str, Type[Agent]]]
    ) -> "LocalAgentRegistry":
        agent_definitions = {}
        for key, (path, klass) in data.items():
            with open(path, "r") as fp:
                agent_definitions[key] = (klass, yaml.safe_load(fp))

        return cls(agent_definitions)
