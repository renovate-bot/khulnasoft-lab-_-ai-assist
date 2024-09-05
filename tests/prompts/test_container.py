import sys
from pathlib import Path
from typing import cast

import pytest
from dependency_injector import containers, providers
from pydantic_core import Url

from ai_gateway.chat.agents.react import ReActAgent
from ai_gateway.config import Config
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.registry import LocalPromptRegistry
from ai_gateway.prompts.typing import ModelMetadata


@pytest.fixture
def mock_config():
    yield Config(custom_models={"enabled": True})


def test_container(mock_container: containers.DeclarativeContainer):
    prompts = cast(providers.Container, mock_container.pkg_prompts)
    registry = cast(LocalPromptRegistry, prompts.prompt_registry())

    prompts_dir = Path(
        sys.modules[LocalPromptRegistry.__module__].__file__ or ""
    ).parent
    prompts_definitions_dir = prompts_dir / "definitions"
    # Iterate over every file in the prompts definitions directory. Make sure
    # they're loaded into the registry and that the resulting Prompts are valid.
    for path in prompts_definitions_dir.glob("**/*.yml"):
        prompt_id_with_model_name = path.relative_to(
            prompts_definitions_dir
        ).with_suffix("")
        prompt_id = prompt_id_with_model_name.parent
        model_name = prompt_id_with_model_name.name

        if model_name == "base":
            model_metadata = None
        else:
            model_metadata = ModelMetadata(
                name=str(model_name), endpoint=Url("http://localhost:4000"), provider=""
            )

        prompt = registry.get(str(prompt_id), model_metadata=model_metadata)
        assert isinstance(prompt, Prompt)

        if isinstance(prompt, ReActAgent):
            prompt_template = prompt.bound.middle[0]
        else:
            prompt_template = prompt.bound.first

        # Check that the messages are populated correctly
        assert len(prompt_template.format_messages()) > 0
