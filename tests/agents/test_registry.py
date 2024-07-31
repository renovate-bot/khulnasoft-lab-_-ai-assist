from pathlib import Path
from typing import Sequence, Type, cast

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import RunnableBinding, RunnableSequence
from pydantic import BaseModel, HttpUrl
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.agents import Agent, AgentRegistered, LocalAgentRegistry
from ai_gateway.agents.config import (
    AgentConfig,
    ChatAnthropicParams,
    ChatLiteLLMParams,
    ModelClassProvider,
    ModelConfig,
)
from ai_gateway.agents.typing import ModelMetadata, TypeModelFactory


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
  params:
    model_class_provider: litellm
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
    custom_llm_provider: vllm
unit_primitives:
  - explain_code
prompt_template:
  system: Template1
""",
    )
    fs.create_file(
        agents_definitions_dir / "chat" / "react" / "base.yml",
        contents="""
---
name: Chat react agent
model:
  name: claude-3-haiku-20240307
  params:
    model_class_provider: anthropic
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
    default_headers:
      header1: "Header1 value"
      header2: "Header2 value"
unit_primitives:
  - duo_chat
prompt_template:
  system: Template1
  user: Template2
params:
  timeout: 60
  stop:
    - Foo
    - Bar
""",
    )
    fs.create_file(
        agents_definitions_dir / "chat" / "react" / "custom.yml",
        contents="""
---
name: Chat react custom agent
model:
  name: custom
  params:
    model_class_provider: litellm
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
unit_primitives:
  - duo_chat
prompt_template:
  system: Template1
  user: Template2
params:
  timeout: 60
  stop:
    - Foo
    - Bar
""",
    )
    yield fs


@pytest.fixture
def model_factories():
    yield {
        ModelClassProvider.ANTHROPIC: lambda model, **kwargs: ChatAnthropic(model=model, **kwargs),  # type: ignore[call-arg]
        ModelClassProvider.LITE_LLM: lambda model, **kwargs: ChatLiteLLM(
            model=model, **kwargs
        ),
    }


@pytest.fixture
def agents_registered():
    yield {
        "test/base": AgentRegistered(
            klass=Agent,
            config=AgentConfig(
                name="Test agent",
                model=ModelConfig(
                    name="claude-2.1",
                    params=ChatLiteLLMParams(
                        model_class_provider=ModelClassProvider.LITE_LLM,
                        top_p=0.1,
                        top_k=50,
                        max_tokens=256,
                        max_retries=10,
                        custom_llm_provider="vllm",
                    ),
                ),
                unit_primitives=["explain_code"],
                prompt_template={"system": "Template1"},
            ),
        ),
        "chat/react/base": AgentRegistered(
            klass=MockAgentClass,
            config=AgentConfig(
                name="Chat react agent",
                model=ModelConfig(
                    name="claude-3-haiku-20240307",
                    provider="anthropic",
                    params=ChatAnthropicParams(
                        model_class_provider=ModelClassProvider.ANTHROPIC,
                        temperature=0.1,
                        top_p=0.8,
                        top_k=40,
                        max_tokens=256,
                        max_retries=6,
                        default_headers={
                            "header1": "Header1 value",
                            "header2": "Header2 value",
                        },
                    ),
                ),
                unit_primitives=["duo_chat"],
                prompt_template={"system": "Template1", "user": "Template2"},
                params={"timeout": 60, "stop": ["Foo", "Bar"]},
            ),
        ),
        "chat/react/custom": AgentRegistered(
            klass=MockAgentClass,
            config=AgentConfig(
                name="Chat react custom agent",
                model=ModelConfig(
                    name="custom",
                    provider="litellm",
                    params=ChatLiteLLMParams(
                        model_class_provider=ModelClassProvider.LITE_LLM,
                        temperature=0.1,
                        top_p=0.8,
                        top_k=40,
                        max_tokens=256,
                        max_retries=6,
                    ),
                ),
                unit_primitives=["duo_chat"],
                prompt_template={"system": "Template1", "user": "Template2"},
                params={"timeout": 60, "stop": ["Foo", "Bar"]},
            ),
        ),
    }


@pytest.fixture
def custom_models_enabled():
    yield True


@pytest.fixture
def registry(
    agents_registered: dict[str, AgentRegistered],
    model_factories: dict[ModelClassProvider, TypeModelFactory],
    custom_models_enabled: bool,
):
    yield LocalAgentRegistry(
        model_factories=model_factories,
        agents_registered=agents_registered,
        custom_models_enabled=custom_models_enabled,
    )


class TestLocalAgentRegistry:
    def test_from_local_yaml(
        self,
        mock_fs: FakeFilesystem,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        agents_registered: dict[str, AgentRegistered],
    ):
        registry = LocalAgentRegistry.from_local_yaml(
            class_overrides={
                "chat/react": MockAgentClass,
            },
            model_factories=model_factories,
            custom_models_enabled=False,
        )

        assert registry.agents_registered == agents_registered

    @pytest.mark.parametrize(
        (
            "agent_id",
            "model_metadata",
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
                None,
                "Test agent",
                Agent,
                [("system", "Template1")],
                "claude-2.1",
                {},
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                    "custom_llm_provider": "vllm",
                },
            ),
            (
                "chat/react",
                None,
                "Chat react agent",
                MockAgentClass,
                [("system", "Template1"), ("user", "Template2")],
                "claude-3-haiku-20240307",
                {"stop": ["Foo", "Bar"], "timeout": 60},
                {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_tokens": 256,
                    "max_retries": 6,
                    "default_headers": {
                        "header1": "Header1 value",
                        "header2": "Header2 value",
                    },
                },
            ),
            (
                "chat/react",
                ModelMetadata(
                    name="custom",
                    endpoint=HttpUrl("http://localhost:4000/"),
                    api_key="token",
                    provider="openai",
                ),
                "Chat react custom agent",
                MockAgentClass,
                [("system", "Template1"), ("user", "Template2")],
                "custom",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "model": "custom",
                    "custom_llm_provider": "openai",
                    "api_key": "token",
                    "api_base": "http://localhost:4000/",
                },
                {
                    "temperature": 0.1,
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
        registry: LocalAgentRegistry,
        agent_id: str,
        model_metadata: ModelMetadata | None,
        expected_name: str,
        expected_class: Type[Agent],
        expected_messages: Sequence[MessageLikeRepresentation],
        expected_model: str,
        expected_kwargs: dict,
        expected_model_params: dict,
    ):

        agent = registry.get(agent_id, {}, model_metadata)

        chain = cast(RunnableSequence, agent.bound)
        actual_messages = cast(ChatPromptTemplate, chain.first).messages
        binding = cast(RunnableBinding, chain.last)
        actual_model = cast(BaseModel, binding.bound)

        assert agent.name == expected_name
        assert isinstance(agent, expected_class)
        assert (
            actual_messages
            == ChatPromptTemplate.from_messages(expected_messages).messages
        )
        assert getattr(binding.bound, "model") == expected_model
        assert binding.kwargs == expected_kwargs

        actual_model_params = {
            key: value
            for key, value in dict(actual_model).items()
            if key in expected_model_params
        }
        assert actual_model_params == expected_model_params

    @pytest.mark.parametrize("custom_models_enabled", [False])
    def test_invalid_get(
        self, registry: LocalAgentRegistry, custom_models_enabled: bool
    ):
        model_metadata = ModelMetadata(
            name="custom",
            endpoint=HttpUrl("http://localhost:4000/"),
            api_key="token",
            provider="openai",
        )

        with pytest.raises(
            ValueError,
            match="Endpoint override not allowed when custom models are disabled.",
        ):
            registry.get("chat/react", {}, model_metadata)
