from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, Optional, Protocol, Type

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ai_gateway.agents.base import Agent, BaseAgentRegistry

__all__ = ["LocalAgentRegistry", "ModelProvider"]


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"


class ModelFactoryType(Protocol):
    def __call__(
        self, *, model: str, **model_kwargs: Optional[Any]
    ) -> BaseChatModel: ...


class AgentRegistered(NamedTuple):
    klass: Type[Agent]
    config: dict


class LocalAgentRegistry(BaseAgentRegistry):
    key_agent_type_base: str = "base"

    def __init__(
        self,
        agents_registered: dict[str, AgentRegistered],
        model_factories: dict[ModelProvider, ModelFactoryType],
    ):
        self.agents_registered = agents_registered
        self.model_factories = model_factories

    def _resolve_id(self, agent_id: str) -> str:
        _, _, agent_type = agent_id.partition("/")
        if agent_type:
            # the `agent_id` value is already in the format of - `first/last`
            return agent_id

        return f"{agent_id}/{self.key_agent_type_base}"

    def _get_model(
        self, provider: str, name: str, **kwargs: Optional[Any]
    ) -> BaseChatModel:
        if model_factory := self.model_factories.get(ModelProvider(provider), None):
            return model_factory(model=name, **kwargs)

        raise ValueError(f"unknown provider `{provider}`.")

    def get(self, agent_id: str, options: Optional[dict[str, Any]] = None) -> Any:
        agent_id = self._resolve_id(agent_id)
        klass, config = self.agents_registered[agent_id]

        # TODO: read model parameters such as `temperature`, `top_k`
        #  and pass them to the model factory via **kwargs.
        model: Runnable = self._get_model(
            provider=config["provider"],
            name=config["model"],
        )

        if "stop" in config:
            model = model.bind(stop=config["stop"])

        messages = klass.build_messages(config["prompt_template"], options or {})
        prompt = ChatPromptTemplate.from_messages(messages)

        return klass(
            name=config["name"],
            chain=prompt | model,
            unit_primitives=config["unit_primitives"],
        )

    @classmethod
    def from_local_yaml(
        cls,
        model_factories: dict[ModelProvider, ModelFactoryType],
        class_overrides: dict[str, Type[Agent]],
    ) -> "LocalAgentRegistry":
        """Iterate over all agent definition files matching [usecase]/[type].yml,
        and create a corresponding agent for each one. The base Agent class is
        used if no matching override is provided in `class_overrides`.
        """

        agents_definitions_dir = Path(__file__).parent / "definitions"
        agents_registered = {}
        for path in agents_definitions_dir.glob("*/*.yml"):
            agent_id = str(
                # E.g., "chat/react", "generate_description/base", etc.
                path.relative_to(agents_definitions_dir).with_suffix("")
            )

            with open(path, "r") as fp:
                klass = class_overrides.get(agent_id, Agent)
                agents_registered[agent_id] = AgentRegistered(
                    klass=klass, config=yaml.safe_load(fp)
                )

        return cls(agents_registered, model_factories)
