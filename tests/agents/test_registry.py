import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import pytest
from jinja2 import BaseLoader, Environment

from ai_gateway import agents
from ai_gateway.agents import LocalAgentRegistry


class TestLocalAgentRegistry:
    def test_get(self):
        registry = LocalAgentRegistry()
        agent_yml = """
---
name: Test agent
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
            assert agent.prompt_templates == {"foo": "Template1", "bar": "Template2"}
