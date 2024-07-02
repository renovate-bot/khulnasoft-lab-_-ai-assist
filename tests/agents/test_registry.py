from pathlib import Path
from typing import Sequence, Type

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessageLikeRepresentation
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.agents import (
    Agent,
    AgentConfig,
    AgentRegistered,
    LocalAgentRegistry,
    Model,
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
model:
  name: claude-2.1
  provider: anthropic
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
model: 
  name: claude-3-haiku-20240307
  provider: anthropic
  params:
    temperature: 0.1
    timeout: 60
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
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
            config=AgentConfig(
                name="Test agent",
                model=Model(name="claude-2.1", provider="anthropic"),
                unit_primitives=["explain_code"],
                prompt_template={"system": "Template1"},
            ),
        ),
        "chat/react": AgentRegistered(
            klass=MockAgentClass,
            config=AgentConfig(
                name="Chat react agent",
                model=Model(
                    name="claude-3-haiku-20240307",
                    provider="anthropic",
                    params=Model.Params(
                        temperature=0.1,
                        timeout=60,
                        top_p=0.8,
                        top_k=40,
                        max_tokens=256,
                        max_retries=6,
                    ),
                ),
                unit_primitives=["duo_chat"],
                prompt_template={"system": "Template1", "user": "Template2"},
                stop=["Foo", "Bar"],
            ),
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
            "expected_model_params",
        ),
        [
            (
                "test",
                "Test agent",
                Agent,
                [("system", "Template1")],
                "claude-2.1",
                None,
                None,
            ),
            (
                "test/base",
                "Test agent",
                Agent,
                [("system", "Template1")],
                "claude-2.1",
                None,
                None,
            ),
            (
                "chat/react",
                "Chat react agent",
                MockAgentClass,
                [("system", "Template1"), ("user", "Template2")],
                "claude-3-haiku-20240307",
                {"stop": ["Foo", "Bar"]},
                {
                    "temperature": 0.1,
                    "request_timeout": 60,  # accessed by alias
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_tokens": 256,
                    "max_retries": 6,
                },
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
        expected_model_params: dict | None,
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

        if expected_model_params:
            actual_model_params = {
                key: value
                for key, value in dict(actual_model.bound).items()
                if key in expected_model_params
            }
            assert actual_model_params == expected_model_params
