from pathlib import Path
from typing import NamedTuple, Optional, Type

import yaml

from ai_gateway.prompts.base import BasePromptRegistry, Prompt
from ai_gateway.prompts.config import ModelClassProvider, PromptConfig
from ai_gateway.prompts.typing import ModelMetadata, TypeModelFactory

__all__ = ["LocalPromptRegistry", "PromptRegistered"]


class PromptRegistered(NamedTuple):
    klass: Type[Prompt]
    config: PromptConfig


class LocalPromptRegistry(BasePromptRegistry):
    key_prompt_type_base: str = "base"

    def __init__(
        self,
        prompts_registered: dict[str, PromptRegistered],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        custom_models_enabled: bool,
    ):
        self.prompts_registered = prompts_registered
        self.model_factories = model_factories
        self.custom_models_enabled = custom_models_enabled

    def _resolve_id(
        self,
        prompt_id: str,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> str:
        if model_metadata:
            return f"{prompt_id}/{model_metadata.name}"

        return f"{prompt_id}/{self.key_prompt_type_base}"

    def get(
        self,
        prompt_id: str,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> Prompt:
        if (
            model_metadata
            and model_metadata.endpoint
            and not self.custom_models_enabled
        ):
            raise ValueError(
                "Endpoint override not allowed when custom models are disabled."
            )

        prompt_id = self._resolve_id(prompt_id, model_metadata)
        klass, config = self.prompts_registered[prompt_id]
        model_class_provider = config.model.params.model_class_provider
        model_factory = self.model_factories.get(model_class_provider, None)

        if not model_factory:
            raise ValueError(
                f"unrecognized model class provider `{model_class_provider}`."
            )

        return klass(model_factory, config, model_metadata)

    @classmethod
    def from_local_yaml(
        cls,
        class_overrides: dict[str, Type[Prompt]],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        custom_models_enabled: bool = False,
    ) -> "LocalPromptRegistry":
        """Iterate over all prompt definition files matching [usecase]/[type].yml,
        and create a corresponding prompt for each one. The base Prompt class is
        used if no matching override is provided in `class_overrides`.
        """

        prompts_definitions_dir = Path(__file__).parent / "definitions"
        prompts_registered = {}

        for path in prompts_definitions_dir.glob("**/*.yml"):
            # E.g., "chat/react/base", "generate_description/mistral", etc.
            prompt_id_with_model_name = path.relative_to(
                prompts_definitions_dir
            ).with_suffix("")

            with open(path, "r") as fp:
                # Remove model name, for example: to receive "chat/react" from "chat/react/mistral"
                klass = class_overrides.get(
                    str(prompt_id_with_model_name.parent), Prompt
                )
                prompts_registered[str(prompt_id_with_model_name)] = PromptRegistered(
                    klass=klass, config=PromptConfig(**yaml.safe_load(fp))
                )

        return cls(prompts_registered, model_factories, custom_models_enabled)
