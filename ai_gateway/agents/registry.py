from pathlib import Path
from typing import Any, NamedTuple, Optional, Type

import yaml

from ai_gateway.agents.base import Agent, BaseAgentRegistry
from ai_gateway.agents.config import AgentConfig, ModelClassProvider
from ai_gateway.agents.typing import ModelMetadata, TypeModelFactory

__all__ = ["LocalAgentRegistry", "AgentRegistered"]


class AgentRegistered(NamedTuple):
    klass: Type[Agent]
    config: AgentConfig


class LocalAgentRegistry(BaseAgentRegistry):
    key_agent_type_base: str = "base"

    def __init__(
        self,
        agents_registered: dict[str, AgentRegistered],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        custom_models_enabled: bool,
    ):
        self.agents_registered = agents_registered
        self.model_factories = model_factories
        self.custom_models_enabled = custom_models_enabled

    def _resolve_id(
        self,
        agent_id: str,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> str:
        if model_metadata:
            return f"{agent_id}/{model_metadata.name}"

        return f"{agent_id}/{self.key_agent_type_base}"

    def get(
        self,
        agent_id: str,
        options: Optional[dict[str, Any]] = None,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Agent:
        if (
            model_metadata
            and model_metadata.endpoint
            and not self.custom_models_enabled
        ):
            raise ValueError(
                "Endpoint override not allowed when custom models are disabled."
            )

        agent_id = self._resolve_id(agent_id, model_metadata)
        klass, config = self.agents_registered[agent_id]
        model_class_provider = config.model.params.model_class_provider
        model_factory = self.model_factories.get(model_class_provider, None)

        if not model_factory:
            raise ValueError(
                f"unrecognized model class provider `{model_class_provider}`."
            )

        return klass(model_factory, config, model_metadata, options)

    @classmethod
    def from_local_yaml(
        cls,
        class_overrides: dict[str, Type[Agent]],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        custom_models_enabled: bool = False,
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

        return cls(agents_registered, model_factories, custom_models_enabled)
