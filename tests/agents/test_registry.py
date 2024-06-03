from pathlib import Path
from unittest.mock import mock_open, patch

from langchain_core.prompts import ChatPromptTemplate

from ai_gateway import agents
from ai_gateway.agents.base import Agent
from ai_gateway.agents.registry import Key, LocalAgentRegistry


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
                {Key(use_case="chat", type="test"): Agent}
            )

            agent = registry.get("chat", "test")

            chain = agent.bound
            expected_messages = ChatPromptTemplate.from_messages(
                [("system", "Template1"), ("user", "Template2")]
            ).messages
            actual_messages = chain.first.messages
            actual_model = chain.last

            mock_file.assert_called_with(
                Path(agents.__file__).parent / "chat" / "test.yml", "r"
            )

            assert actual_messages == expected_messages
            assert actual_model.model == "claude-3-haiku-20240307"
            assert actual_model.kwargs == {"stop": ["Foo", "Bar"]}
            assert agent.name == "Test agent"
