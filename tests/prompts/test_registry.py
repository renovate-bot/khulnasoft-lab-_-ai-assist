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

from ai_gateway.prompts import LocalPromptRegistry, Prompt, PromptRegistered
from ai_gateway.prompts.config import (
    ChatAnthropicParams,
    ChatLiteLLMParams,
    ModelClassProvider,
    ModelConfig,
    PromptConfig,
)
from ai_gateway.prompts.typing import ModelMetadata, TypeModelFactory


class MockPromptClass(Prompt):
    pass


@pytest.fixture
def mock_fs(fs: FakeFilesystem):
    prompts_definitions_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "prompts" / "definitions"
    )
    fs.create_file(
        prompts_definitions_dir / "test" / "base.yml",
        contents="""
---
name: Test prompt
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
        prompts_definitions_dir / "chat" / "react" / "base.yml",
        contents="""
---
name: Chat react prompt
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
        prompts_definitions_dir / "chat" / "react" / "custom.yml",
        contents="""
---
name: Chat react custom prompt
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
  vertex_location: us-east1
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
def prompts_registered():
    yield {
        "test/base": PromptRegistered(
            klass=Prompt,
            config=PromptConfig(
                name="Test prompt",
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
        "chat/react/base": PromptRegistered(
            klass=MockPromptClass,
            config=PromptConfig(
                name="Chat react prompt",
                model=ModelConfig(
                    name="claude-3-haiku-20240307",
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
        "chat/react/custom": PromptRegistered(
            klass=MockPromptClass,
            config=PromptConfig(
                name="Chat react custom prompt",
                model=ModelConfig(
                    name="custom",
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
                params={
                    "timeout": 60,
                    "stop": ["Foo", "Bar"],
                    "vertex_location": "us-east1",
                },
            ),
        ),
    }


@pytest.fixture
def default_prompts():
    yield {}


@pytest.fixture
def custom_models_enabled():
    yield True


@pytest.fixture
def registry(
    prompts_registered: dict[str, PromptRegistered],
    model_factories: dict[ModelClassProvider, TypeModelFactory],
    default_prompts: dict[str, str],
    custom_models_enabled: bool,
):
    yield LocalPromptRegistry(
        model_factories=model_factories,
        prompts_registered=prompts_registered,
        default_prompts=default_prompts,
        custom_models_enabled=custom_models_enabled,
    )


class TestLocalPromptRegistry:
    def test_from_local_yaml(
        self,
        mock_fs: FakeFilesystem,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        prompts_registered: dict[str, PromptRegistered],
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            class_overrides={
                "chat/react": MockPromptClass,
            },
            model_factories=model_factories,
            default_prompts={},
            custom_models_enabled=False,
        )

        assert registry.prompts_registered == prompts_registered

    @pytest.mark.parametrize(
        (
            "prompt_id",
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
                "Test prompt",
                Prompt,
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
                "Chat react prompt",
                MockPromptClass,
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
                    provider="custom_openai",
                    identifier="anthropic/claude-3-haiku-20240307",
                ),
                "Chat react custom prompt",
                MockPromptClass,
                [("system", "Template1"), ("user", "Template2")],
                "custom",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "model": "claude-3-haiku-20240307",
                    "custom_llm_provider": "anthropic",
                    "api_key": "token",
                    "api_base": "http://localhost:4000/",
                    "vertex_location": "us-east1",
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
        registry: LocalPromptRegistry,
        prompt_id: str,
        model_metadata: ModelMetadata | None,
        expected_name: str,
        expected_class: Type[Prompt],
        expected_messages: Sequence[MessageLikeRepresentation],
        expected_model: str,
        expected_kwargs: dict,
        expected_model_params: dict,
    ):

        prompt = registry.get(prompt_id, model_metadata)

        chain = cast(RunnableSequence, prompt.bound)
        actual_messages = cast(ChatPromptTemplate, chain.first).messages
        binding = cast(RunnableBinding, chain.last)
        actual_model = cast(BaseModel, binding.bound)

        assert prompt.name == expected_name
        assert isinstance(prompt, expected_class)
        assert (
            actual_messages
            == ChatPromptTemplate.from_messages(
                expected_messages, template_format="jinja2"
            ).messages
        )
        assert prompt.model_name == expected_model
        assert binding.kwargs == expected_kwargs

        actual_model_params = {
            key: value
            for key, value in dict(actual_model).items()
            if key in expected_model_params
        }
        assert actual_model_params == expected_model_params

    @pytest.mark.parametrize(
        (
            "prompt_id",
            "model_metadata",
            "expected_name",
            "expected_class",
            "expected_model",
            "expected_kwargs",
        ),
        [
            (
                "code_suggestions/generations",
                None,
                "Claude 3 Code Generations Agent",
                Prompt,
                "claude-3-5-sonnet-20240620",
                {"stop": ["</new_code>"]},
            ),
        ],
    )
    def test_get_code_generations_base(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        prompt_id: str,
        model_metadata: ModelMetadata | None,
        expected_name: str,
        expected_class: Type[Prompt],
        expected_model: str,
        expected_kwargs: dict,
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            class_overrides={},
            model_factories=model_factories,
            default_prompts={},
        )
        prompt = registry.get(prompt_id, model_metadata)
        chain = cast(RunnableSequence, prompt.bound)
        binding = cast(RunnableBinding, chain.last)

        assert prompt.name == expected_name
        assert isinstance(prompt, expected_class)
        assert prompt.model_name == expected_model
        assert binding.kwargs == expected_kwargs

    def test_default_prompts(
        self,
        mock_fs: FakeFilesystem,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        prompts_registered: dict[str, PromptRegistered],
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            class_overrides={
                "chat/react": MockPromptClass,
            },
            model_factories=model_factories,
            default_prompts={"chat/react": "custom"},
            custom_models_enabled=False,
        )

        assert registry.get("chat/react").name == "Chat react custom prompt"

    @pytest.mark.parametrize("custom_models_enabled", [False])
    def test_invalid_get(
        self, registry: LocalPromptRegistry, custom_models_enabled: bool
    ):
        model_metadata = ModelMetadata(
            name="custom",
            endpoint=HttpUrl("http://localhost:4000/"),
            api_key="token",
            provider="custom_openai",
        )

        with pytest.raises(
            ValueError,
            match="Endpoint override not allowed when custom models are disabled.",
        ):
            registry.get("chat/react", model_metadata)
