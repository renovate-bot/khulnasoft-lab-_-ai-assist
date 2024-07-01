from pathlib import Path
from typing import Sequence, Type

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessageLikeRepresentation
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.agents.base import Agent
from ai_gateway.agents.registry import (
    AgentRegistered,
    LocalAgentRegistry,
    ModelFactoryType,
)


class MockAgentClass(Agent):
    pass


@pytest.fixture
def mock_fs(fs: FakeFilesystem):
    agents_definitions_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "agents" / "definitions"
    )
    fs.create_file(
        agents_definitions_dir / "test" / "base.yml",
        contents="""
---
name: Test agent
provider: anthropic
model: claude-2.1
unit_primitives:
  - explain_code
prompt_template:
  system: Template1
""",
    )
    fs.create_file(
        agents_definitions_dir / "chat" / "react.yml",
        contents="""
---
name: Chat react agent
provider: anthropic
model: claude-3-haiku-20240307
unit_primitives:
  - duo_chat
prompt_template:
  system: Template1
  user: Template2
stop:
  - Foo
  - Bar
""",
    )
    yield fs


@pytest.fixture
def agents_registered():
    yield {
        "test/base": AgentRegistered(
            klass=Agent,
            config={
                "name": "Test agent",
                "provider": "anthropic",
                "model": "claude-2.1",
                "unit_primitives": ["explain_code"],
                "prompt_template": {"system": "Template1"},
            },
        ),
        "chat/react": AgentRegistered(
            klass=MockAgentClass,
            config={
                "name": "Chat react agent",
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "unit_primitives": ["duo_chat"],
                "prompt_template": {"system": "Template1", "user": "Template2"},
                "stop": ["Foo", "Bar"],
            },
        ),
    }


class TestLocalAgentRegistry:
    def test_from_local_yaml(
        self,
        mock_fs: FakeFilesystem,
        agents_registered: dict[str, AgentRegistered],
    ):
        registry = LocalAgentRegistry.from_local_yaml({"chat/react": MockAgentClass})

        assert registry.agents_registered == agents_registered

    @pytest.mark.parametrize(
        (
            "agent_id",
            "expected_name",
            "expected_class",
            "expected_messages",
            "expected_model",
            "expected_kwargs",
        ),
        [
            (
                "test",
                "Test agent",
                Agent,
                [("system", "Template1")],
                "claude-2.1",
                None,
            ),
            (
                "test/base",
                "Test agent",
                Agent,
                [("system", "Template1")],
                "claude-2.1",
                None,
            ),
            (
                "chat/react",
                "Chat react agent",
                MockAgentClass,
                [("system", "Template1"), ("user", "Template2")],
                "claude-3-haiku-20240307",
                {"stop": ["Foo", "Bar"]},
            ),
        ],
    )
    def test_get(
        self,
        agents_registered: dict[str, AgentRegistered],
        agent_id: str,
        expected_name: str,
        expected_class: Type[Agent],
        expected_messages: Sequence[MessageLikeRepresentation],
        expected_model: str,
        expected_kwargs: dict,
    ):
        registry = LocalAgentRegistry(
            agents_registered=agents_registered,
        )

        agent = registry.get(agent_id)

        chain = agent.bound
        actual_messages = chain.first.messages
        actual_model = chain.last

        assert agent.name == expected_name
        assert isinstance(agent, expected_class)
        assert (
            actual_messages
            == ChatPromptTemplate.from_messages(expected_messages).messages
        )
        assert actual_model.model == expected_model

        if expected_kwargs:
            assert actual_model.kwargs == expected_kwargs
