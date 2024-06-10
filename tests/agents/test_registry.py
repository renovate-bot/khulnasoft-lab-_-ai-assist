from pathlib import Path
from unittest.mock import mock_open, patch

from anthropic import AsyncAnthropic
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway import agents
from ai_gateway.agents.base import Agent
from ai_gateway.agents.registry import Key, LocalAgentRegistry, ModelProvider
from ai_gateway.models.v2 import ChatAnthropic


class TestLocalAgentRegistry:
    def test_get(self):
        agent_yml = """
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
        """

        with patch("builtins.open", mock_open(read_data=agent_yml)) as mock_file:
            registry = LocalAgentRegistry.from_local_yaml(
                data={Key(use_case="chat", type="test"): Agent},
                model_factories={
                    ModelProvider.ANTHROPIC: lambda model: ChatAnthropic(
                        async_client=AsyncAnthropic(), model=model
                    )
                },
            )

            agent = registry.get("chat", "test")

            chain = agent.bound
            actual_messages = chain.first.messages
            actual_model = chain.last

            expected_messages = ChatPromptTemplate.from_messages(
                [("system", "Template1"), ("user", "Template2")]
            ).messages

            mock_file.assert_called_with(
                Path(agents.__file__).parent / "chat" / "test.yml", "r"
            )

            assert actual_messages == expected_messages
            assert actual_model.model == "claude-3-haiku-20240307"
            assert actual_model.kwargs == {"stop": ["Foo", "Bar"]}
            assert agent.name == "Test agent"
