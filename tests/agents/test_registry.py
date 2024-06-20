from pathlib import Path
from typing import Type
from unittest.mock import mock_open, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway import agents
from ai_gateway.agents.base import Agent
from ai_gateway.agents.registry import Key, LocalAgentRegistry, ModelProvider


class MockAgentClass(Agent):
    pass


class TestLocalAgentRegistry:
    @pytest.mark.parametrize(
        ("agent_yml", "class_overrides", "expected_class", "expected_kwargs"),
        [
            (
                """
---
name: Test agent
provider: anthropic
model: claude-3-haiku-20240307
prompt_template:
  system: Template1
  user: Template2
            """,
                {},
                Agent,
                None,
            ),
            (
                """
---
name: Test agent
provider: anthropic
model: claude-3-haiku-20240307
prompt_template:
  system: Template1
  user: Template2
stop:
  - Foo
  - Bar
            """,
                {Key(use_case="chat", type="react"): MockAgentClass},
                MockAgentClass,
                {"stop": ["Foo", "Bar"]},
            ),
        ],
    )
    def test_get(
        self,
        agent_yml: str,
        class_overrides: dict[Key, Type[Agent]],
        expected_class: Type[Agent],
        expected_kwargs: dict,
    ):

        with patch("builtins.open", mock_open(read_data=agent_yml)) as mock_file:
            registry = LocalAgentRegistry.from_local_yaml(
                class_overrides=class_overrides,
                model_factories={
                    ModelProvider.ANTHROPIC: lambda model, **model_kwargs: ChatAnthropic(model=model)  # type: ignore[call-arg]
                },
            )

            agent = registry.get("chat", "react")

            chain = agent.bound
            actual_messages = chain.first.messages
            actual_model = chain.last

            expected_messages = ChatPromptTemplate.from_messages(
                [("system", "Template1"), ("user", "Template2")]
            ).messages

            mock_file.assert_called_with(
                Path(agents.__file__).parent / "chat" / "react.yml", "r"
            )

            assert agent.name == "Test agent"
            assert isinstance(agent, expected_class)
            assert actual_messages == expected_messages
            assert actual_model.model == "claude-3-haiku-20240307"

            if expected_kwargs:
                assert actual_model.kwargs == expected_kwargs
