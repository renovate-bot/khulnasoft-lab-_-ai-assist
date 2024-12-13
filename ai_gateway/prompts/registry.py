from pathlib import Path
from typing import NamedTuple, Optional, Type

import structlog
import yaml
from poetry.core.constraints.version import Version, parse_constraint

from ai_gateway.prompts.base import BasePromptRegistry, Prompt
from ai_gateway.prompts.config import ModelClassProvider, PromptConfig
from ai_gateway.prompts.typing import ModelMetadata, TypeModelFactory

__all__ = ["LocalPromptRegistry", "PromptRegistered"]

log = structlog.stdlib.get_logger("prompts")


class PromptRegistered(NamedTuple):
    klass: Type[Prompt]
    versions: dict[str, PromptConfig]


class LocalPromptRegistry(BasePromptRegistry):
    key_prompt_type_base: str = "base"

    def __init__(
        self,
        prompts_registered: dict[str, PromptRegistered],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        default_prompts: dict[str, str],
        custom_models_enabled: bool,
        disable_streaming: bool = False,
    ):
        self.prompts_registered = prompts_registered
        self.model_factories = model_factories
        self.default_prompts = default_prompts
        self.custom_models_enabled = custom_models_enabled
        self.disable_streaming = disable_streaming

    def _resolve_id(
        self,
        prompt_id: str,
        model_metadata: Optional[ModelMetadata] = None,
    ) -> str:
        if model_metadata:
            return f"{prompt_id}/{model_metadata.name}"

        type = self.default_prompts.get(prompt_id, self.key_prompt_type_base)
        return f"{prompt_id}/{type}"

    def _get_prompt_config(
        self, versions: dict[str, PromptConfig], prompt_version: str
    ) -> PromptConfig:
        constraint = parse_constraint(prompt_version)
        all_versions = [Version.parse(version) for version in versions.keys()]
        compatible_versions = list(filter(constraint.allows, all_versions))
        compatible_versions.sort(reverse=True)

        return versions[str(compatible_versions[0])]

    def get(
        self,
        prompt_id: str,
        prompt_version: str,
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
        prompt_registered = self.prompts_registered[prompt_id]
        config = self._get_prompt_config(prompt_registered.versions, prompt_version)
        model_class_provider = config.model.params.model_class_provider
        model_factory = self.model_factories.get(model_class_provider, None)

        if not model_factory:
            raise ValueError(
                f"unrecognized model class provider `{model_class_provider}`."
            )

        log.debug(
            "Returning prompt from the registry",
            prompt_id=prompt_id,
            prompt_name=config.name,
            prompt_version=prompt_version,
        )

        return prompt_registered.klass(
            model_factory,
            config,
            model_metadata,
            disable_streaming=self.disable_streaming,
        )

    @classmethod
    def from_local_yaml(
        cls,
        class_overrides: dict[str, Type[Prompt]],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        default_prompts: dict[str, str],
        custom_models_enabled: bool = False,
        disable_streaming: bool = False,
    ) -> "LocalPromptRegistry":
        """Iterate over all prompt definition files matching [usecase]/[type]/[version].yml,
        and create a corresponding prompt for each one. The base Prompt class is
        used if no matching override is provided in `class_overrides`.
        """

        prompts_definitions_dir = Path(__file__).parent / "definitions"
        prompts_registered = {}

        # Iterate over each folder
        for path in prompts_definitions_dir.glob("**"):
            versions = {}

            # Iterate over each version file
            for version in path.glob("*.yml"):
                with open(version, "r") as fp:
                    versions[version.stem] = PromptConfig(**yaml.safe_load(fp))

            # If there were no yml files in this folder, skip it
            if not versions:
                continue

            # E.g., "chat/react/base", "generate_description/mistral", etc.
            prompt_id_with_model_name = path.relative_to(prompts_definitions_dir)

            klass = class_overrides.get(str(prompt_id_with_model_name.parent), Prompt)
            prompts_registered[str(prompt_id_with_model_name)] = PromptRegistered(
                klass=klass, versions=versions
            )

        log.info(
            "Initializing prompt registry from local yaml",
            default_prompts=default_prompts,
            custom_models_enabled=custom_models_enabled,
        )

        return cls(
            prompts_registered,
            model_factories,
            default_prompts,
            custom_models_enabled,
            disable_streaming,
        )
