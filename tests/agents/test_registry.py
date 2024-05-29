from pathlib import Path
from unittest.mock import mock_open, patch

from ai_gateway import agents
from ai_gateway.agents.chat import ReActAgent
from ai_gateway.agents.registry import Key, LocalAgentRegistry


class TestLocalAgentRegistry:
    def test_get(self):
        agent_yml = """
---
name: Test agent
provider: anthropic
model: claude-3-haiku-20240307
prompt_templates:
  system: Template1
  user: Template2
stop:
    - Foo
    - Bar
        """

        with patch("builtins.open", mock_open(read_data=agent_yml)) as mock_file:
            registry = LocalAgentRegistry.from_local_yaml(
                {
                    Key(use_case="chat", type="test"): (
                        Path(agents.__file__).parent / "chat" / "test.yml",
                        ReActAgent,
                    )
                }
            )

            agent = registry.get("chat", "test")

            chain_data = agent.chain.dict()
            actual_messages = {
                m["type"]: m["content"] for m in chain_data["first"]["messages"]
            }
            actual_model = chain_data["middle"][0]

            mock_file.assert_called_with(
                Path(agents.__file__).parent / "chat" / "test.yml", "r"
            )

            assert actual_messages == {"system": "Template1", "human": "Template2"}
            assert actual_model["bound"]["model"] == "claude-3-haiku-20240307"
            assert actual_model["kwargs"] == {"stop": ["Foo", "Bar"]}
            assert agent.name == "Test agent"
