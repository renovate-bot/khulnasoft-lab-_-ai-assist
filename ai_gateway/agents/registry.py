from pathlib import Path
from typing import Any, NamedTuple, Optional, Protocol, Type

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ai_gateway.agents.base import Agent, BaseAgentRegistry
from ai_gateway.agents.config import AgentConfig, ModelClassProvider, ModelConfig
from ai_gateway.agents.typing import ModelMetadata

__all__ = ["LocalAgentRegistry", "AgentRegistered", "CustomModelsAgentRegistry"]


class TypeModelFactory(Protocol):
    def __call__(self, *, model: str, **kwargs: Optional[Any]) -> BaseChatModel: ...


class AgentRegistered(NamedTuple):
    klass: Type[Agent]
    config: AgentConfig


class LocalAgentRegistry(BaseAgentRegistry):
    key_agent_type_base: str = "base"

    def __init__(
        self,
        agents_registered: dict[str, AgentRegistered],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
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
        self,
        config_model: ModelConfig,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> BaseChatModel:
        model_class_provider = config_model.params.model_class_provider
        if model_factory := self.model_factories.get(model_class_provider, None):
            return model_factory(
                model=config_model.name,
                **config_model.params.model_dump(
                    exclude={"model_class_provider"}, exclude_none=True, by_alias=True
                ),
            )

        raise ValueError(f"unrecognized model class provider `{model_class_provider}`.")

    def get(
        self,
        agent_id: str,
        options: Optional[dict[str, Any]] = None,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Agent:
        agent_id = self._resolve_id(agent_id)
        klass, config = self.agents_registered[agent_id]

        model: Runnable = self._get_model(config.model, model_metadata)

        if config.stop:
            model = model.bind(stop=config.stop)

        messages = klass.build_messages(config.prompt_template, options or {})
        prompt = ChatPromptTemplate.from_messages(messages)

        return klass(
            name=config.name,
            chain=prompt | model,
            unit_primitives=config.unit_primitives,
        )

    @classmethod
    def from_local_yaml(
        cls,
        class_overrides: dict[str, Type[Agent]],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
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
                    klass=klass, config=AgentConfig(**yaml.safe_load(fp))
                )

        return cls(agents_registered, model_factories)


class CustomModelsAgentRegistry(LocalAgentRegistry):
    def _get_model(
        self,
        config_model: ModelConfig,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> BaseChatModel:
        chat_model = super()._get_model(config_model)

        if model_metadata is None:
            return chat_model

        return chat_model.bind(
            model=model_metadata.name,
            api_base=str(model_metadata.endpoint),
            custom_llm_provider=model_metadata.provider,
            api_key=model_metadata.api_key,
        )

    def get(
        self,
        agent_id: str,
        options: Optional[dict[str, Any]] = None,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Agent:
        if model_metadata is not None:
            agent_id = f"{agent_id}-custom"

        return super().get(agent_id, options, model_metadata)
