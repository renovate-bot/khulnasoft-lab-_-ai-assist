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

    def _resolve_id(
        self,
        agent_id: str,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> str:
        if model_metadata:
            return f"{agent_id}/{model_metadata.name}"

        return f"{agent_id}/{self.key_agent_type_base}"

    def _get_model(
        self,
        config_model: ModelConfig,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Runnable:
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
        agent_id = self._resolve_id(agent_id, model_metadata)
        klass, config = self.agents_registered[agent_id]

        model = self._get_model(config.model, model_metadata)

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

        for path in agents_definitions_dir.glob("**/*.yml"):
            agent_id_with_model_name = str(
                # E.g., "chat/react/base", "generate_description/mistral", etc.
                path.relative_to(agents_definitions_dir).with_suffix("")
            )

            # Remove model name, for example: to receive "chat/react" from "chat/react/mistral"
            agent_id, _, _ = agent_id_with_model_name.rpartition("/")

            with open(path, "r") as fp:
                klass = class_overrides.get(agent_id, Agent)
                agents_registered[agent_id_with_model_name] = AgentRegistered(
                    klass=klass, config=AgentConfig(**yaml.safe_load(fp))
                )

        return cls(agents_registered, model_factories)


class CustomModelsAgentRegistry(LocalAgentRegistry):
    def _get_model(
        self,
        config_model: ModelConfig,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Runnable:
        chat_model = super()._get_model(config_model)

        if model_metadata is None:
            return chat_model

        return chat_model.bind(
            model=model_metadata.name,
            api_base=str(model_metadata.endpoint),
            custom_llm_provider=model_metadata.provider,
            api_key=model_metadata.api_key,
        )
