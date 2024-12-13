import sys
from pathlib import Path
from typing import cast

import pytest
from dependency_injector import containers, providers
from pydantic import AnyUrl

from ai_gateway.chat.agents.react import ReActAgent
from ai_gateway.config import Config
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.registry import LocalPromptRegistry
from ai_gateway.prompts.typing import ModelMetadata


@pytest.fixture
def mock_config():
    yield Config(custom_models={"enabled": True, "disable_streaming": True})


def test_container(mock_container: containers.DeclarativeContainer):
    prompts = cast(providers.Container, mock_container.pkg_prompts)
    registry = cast(LocalPromptRegistry, prompts.prompt_registry())

    prompts_dir = Path(
        sys.modules[LocalPromptRegistry.__module__].__file__ or ""
    ).parent
    prompts_definitions_dir = prompts_dir / "definitions"
    # Iterate over every file in the prompts definitions directory. Make sure
    # they're loaded into the registry and that the resulting Prompts are valid.
    for path in prompts_definitions_dir.glob("**"):
        versions = [version.stem for version in path.glob("*.yml")]

        if not versions:
            continue

        prompt_id_with_model_name = path.relative_to(prompts_definitions_dir)
        prompt_id = prompt_id_with_model_name.parent
        model_name = prompt_id_with_model_name.name

        if model_name == "base":
            model_metadata = None
        else:
            model_metadata = ModelMetadata(
                name=str(model_name),
                endpoint=AnyUrl("http://localhost:4000"),
                provider="",
            )

        for version in versions:
            prompt = registry.get(
                str(prompt_id),
                f"={version}",  # Make a strict constraint so we can check every existing version
                model_metadata=model_metadata,
            )
            assert isinstance(prompt, Prompt)
            assert prompt.model.disable_streaming

            if isinstance(prompt, ReActAgent):
                prompt_template = prompt.bound.middle[0]  # type: ignore[attr-defined]
            else:
                prompt_template = prompt.bound.first  # type: ignore[attr-defined]

                # Check that the messages are populated correctly
                assert len(prompt_template.format_messages()) > 0
