from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, Optional, Protocol, Type

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.agents.base import Agent, BaseAgentRegistry

__all__ = ["LocalAgentRegistry", "ModelProvider"]


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"


class ModelFactoryType(Protocol):
    def __call__(
        self, *, model: str, **model_kwargs: Optional[Any]
    ) -> BaseChatModel: ...


class Key(NamedTuple):
    use_case: str
    type: str


class LocalAgentRegistry(BaseAgentRegistry):
    def __init__(
        self,
        agent_definitions: dict[Key, tuple[Type[Agent], dict]],
        model_factories: dict[ModelProvider, ModelFactoryType],
    ):
        self.agent_definitions = agent_definitions
        self.model_factories = model_factories

    def _get_model(
        self, provider: str, name: str, **kwargs: Optional[Any]
    ) -> BaseChatModel:
        if model_factory := self.model_factories.get(ModelProvider(provider), None):
            return model_factory(model=name, **kwargs)

        raise ValueError(f"unknown provider `{provider}`.")

    def get(
        self, use_case: str, agent_type: str, options: Optional[dict[str, Any]] = None
    ) -> Any:
        klass, config = self.agent_definitions[Key(use_case=use_case, type=agent_type)]

        model = self._get_model(
            provider=config["provider"],
            name=config["model"],
            # TODO: read model parameters such as `temperature`, `top_k`
            #  and pass them to the model factory via **kwargs.
        ).bind(stop=config["stop"])

        messages = klass.build_messages(config["prompt_template"], options or {})
        prompt = ChatPromptTemplate.from_messages(messages)

        return klass(
            name=config["name"],
            chain=prompt | model,
        )

    @classmethod
    def from_local_yaml(
        cls,
        data: dict[Key, Type[Agent]],
        model_factories: dict[ModelProvider, ModelFactoryType],
    ) -> "LocalAgentRegistry":
        agent_definitions = {}
        for key, klass in data.items():
            path = Path(__file__).parent / key.use_case / f"{key.type}.yml"
            with open(path, "r") as fp:
                agent_definitions[key] = (klass, yaml.safe_load(fp))

        return cls(agent_definitions, model_factories)
