import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

from ai_gateway import agents
from ai_gateway.agents.registry import LocalAgentRegistry
from ai_gateway.models.base import connect_anthropic


class TestLocalAgentRegistry:
    def test_get(self):
        registry = LocalAgentRegistry(client=connect_anthropic())
        agent_yml = """
---
name: Test agent
model: claude-3-haiku-20240307
prompt_templates:
  foo: Template1
  bar: Template2
        """

        with patch("builtins.open", mock_open(read_data=agent_yml)) as mock_file:
            agent = registry.get("chat", "test")
            mock_file.assert_called_with(
                Path(agents.__file__).parent / "chat" / "test.yml", "r"
            )
            assert agent.name == "Test agent"
            assert agent.model.metadata.name == "claude-3-haiku-20240307"
            assert agent.prompt_templates == {"foo": "Template1", "bar": "Template2"}
