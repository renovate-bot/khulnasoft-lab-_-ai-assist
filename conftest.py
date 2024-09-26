from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Type, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk
from starlette.middleware import Middleware
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.api.middleware import MiddlewareAuthentication, AccessLogMiddleware
from ai_gateway.code_suggestions.base import CodeSuggestionsChunk, CodeSuggestionsOutput
from ai_gateway.code_suggestions.processing.base import ModelEngineOutput
from ai_gateway.code_suggestions.processing.typing import (
    LanguageId,
    MetadataCodeContent,
    MetadataPromptBuilder,
)
from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.experimentation.base import ExperimentTelemetry
from ai_gateway.gitlab_features import GitLabUnitPrimitive
from ai_gateway.models.base import (
    ModelMetadata,
    SafetyAttributes,
    TokensConsumptionMetadata,
)
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig, PromptParams
from ai_gateway.prompts.config.models import ChatLiteLLMParams, TypeModelParams
from ai_gateway.prompts.typing import TypeModelFactory

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def tpl_assets_codegen_dir() -> Path:
    assets_dir = Path(__file__).parent / "tests" / "_assets"
    tpl_dir = assets_dir / "tpl"
    return tpl_dir / "codegen"


@pytest.fixture
def text_gen_base_model():
    model = Mock(spec=TextGenModelBase)
    model.MAX_MODEL_LEN = 1_000
    model.UPPER_BOUND_MODEL_CHARS = model.MAX_MODEL_LEN * 5
    return model


@pytest.fixture(scope="class")
def stub_auth_provider():
    class StubKeyAuthProvider:
        def authenticate(self, token):
            return None

    return StubKeyAuthProvider()


@pytest.fixture(scope="class")
def test_client(fast_api_router, stub_auth_provider, request):
    middlewares = [
        Middleware(RawContextMiddleware),
        Middleware(AccessLogMiddleware, skip_endpoints=[]),
        MiddlewareAuthentication(stub_auth_provider, False, None),
    ]
    app = FastAPI(middleware=middlewares)
    app.include_router(fast_api_router)
    client = TestClient(app)

    return client

@pytest.fixture
def mock_track_internal_event():
    with patch("ai_gateway.internal_events.InternalEventsClient.track_event") as mock:
        yield mock

@pytest.fixture
def mock_client(test_client, stub_auth_provider, auth_user, mock_container):
    """Setup all the needed mocks to perform requests in the test environment
    """
    with patch.object(stub_auth_provider, "authenticate", return_value=auth_user):
        yield test_client


@pytest.fixture
def mock_connect_vertex():
    with patch("ai_gateway.models.base.PredictionServiceAsyncClient"):
        yield


@pytest.fixture
def mock_connect_vertex_search():
    with patch("ai_gateway.searches.container.discoveryengine.SearchServiceAsyncClient"):
        yield


@pytest.fixture
def mock_config():
    yield Config()


@pytest.fixture
def mock_container(mock_config: Config, mock_connect_vertex: Mock, mock_connect_vertex_search: Mock):
    container_application = ContainerApplication()
    container_application.config.from_dict(mock_config.model_dump())

    yield container_application


@pytest.fixture
def mock_output_text():
    yield "test completion"


@pytest.fixture
def mock_output(mock_output_text: str):
    yield TextGenModelOutput(
        text=mock_output_text,
        score=10_000,
        safety_attributes=SafetyAttributes(),
    )


@contextmanager
def _mock_generate(klass: str, mock_output: TextGenModelOutput):
    with patch(f"{klass}.generate", return_value=mock_output) as mock:
        yield mock


@contextmanager
def _mock_async_generate(klass: str, mock_output: TextGenModelOutput):
    async def _stream(*args: Any, **kwargs: Any) -> AsyncIterator[TextGenModelChunk]:
        for c in list(mock_output.text):
            yield TextGenModelChunk(text=c)

    with patch(f"{klass}.generate", side_effect=_stream) as mock:
        yield mock


@pytest.fixture
def mock_code_bison(mock_output: CodeSuggestionsOutput):
    with _mock_generate("ai_gateway.models.vertex_text.PalmCodeBisonModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_code_gecko(mock_output: CodeSuggestionsOutput):
    with _mock_generate("ai_gateway.models.vertex_text.PalmCodeGeckoModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_anthropic(mock_output: TextGenModelOutput):
    with _mock_generate("ai_gateway.models.anthropic.AnthropicModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_anthropic_chat(mock_output: TextGenModelOutput):
    with _mock_generate("ai_gateway.models.anthropic.AnthropicChatModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_anthropic_stream(mock_output: TextGenModelOutput):
    with _mock_async_generate("ai_gateway.models.anthropic.AnthropicModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_anthropic_chat_stream(mock_output: TextGenModelOutput):
    with _mock_async_generate("ai_gateway.models.anthropic.AnthropicChatModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_llm_chat(mock_output: TextGenModelOutput):
    with _mock_generate("ai_gateway.models.litellm.LiteLlmChatModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_llm_text(mock_output: TextGenModelOutput):
    with _mock_generate("ai_gateway.models.litellm.LiteLlmTextGenModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_agent_model(mock_output: TextGenModelOutput):
    with _mock_generate("ai_gateway.models.agent_model.AgentModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_completions_legacy_output_texts():
    yield ["def search"]


@pytest.fixture
def mock_completions_legacy_output(mock_completions_legacy_output_texts: str):
    output = []
    for text in mock_completions_legacy_output_texts:
        output.append(
            ModelEngineOutput(
                text=text,
                score=0,
                model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                lang_id=LanguageId.PYTHON,
                metadata=MetadataPromptBuilder(
                    components={
                        "prefix": MetadataCodeContent(length=10, length_tokens=2),
                        "suffix": MetadataCodeContent(length=10, length_tokens=2),
                    },
                    experiments=[
                        ExperimentTelemetry(name="truncate_suffix", variant=1)
                    ],

                ),
                tokens_consumption_metadata=TokensConsumptionMetadata(
                    input_tokens=1, output_tokens=2
                ),
            )
        )

    yield output

@pytest.fixture
def mock_suggestions_output_text():
    yield "def search"


@pytest.fixture
def mock_suggestions_model():
    yield "claude-instant-1.2"


@pytest.fixture
def mock_suggestions_engine():
    yield "anthropic"


@pytest.fixture
def mock_suggestions_output(mock_suggestions_output_text: str, mock_suggestions_model: str, mock_suggestions_engine: str):
    yield CodeSuggestionsOutput(
        text=mock_suggestions_output_text,
        score=0,
        model=ModelMetadata(name=mock_suggestions_model, engine=mock_suggestions_engine),
        lang_id=LanguageId.PYTHON,
        metadata=CodeSuggestionsOutput.Metadata(experiments=[]),
    )


@pytest.fixture
def mock_completions_legacy(mock_completions_legacy_output: list[ModelEngineOutput]):
    with patch("ai_gateway.code_suggestions.CodeCompletionsLegacy.execute", return_value=mock_completions_legacy_output) as mock:
        yield mock


@contextmanager
def _mock_execute(klass: str, mock_suggestions_output: CodeSuggestionsOutput):
    with patch(f"{klass}.execute", return_value=mock_suggestions_output) as mock:
        yield mock


@pytest.fixture
def mock_generations(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_execute("ai_gateway.code_suggestions.CodeGenerations", mock_suggestions_output) as mock:
        yield mock


@pytest.fixture
def mock_completions(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_execute("ai_gateway.code_suggestions.CodeCompletions", mock_suggestions_output) as mock:
        yield mock


@contextmanager
def _mock_async_execute(klass: str, mock_suggestions_output: CodeSuggestionsOutput):
    async def _stream(*args: Any, **kwargs: Any) -> AsyncIterator[CodeSuggestionsChunk]:
        for c in list(mock_suggestions_output.text):
            yield CodeSuggestionsChunk(text=c)

    with patch(f"{klass}.execute", side_effect=_stream) as mock:
        yield mock


@pytest.fixture
def mock_generations_stream(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_async_execute("ai_gateway.code_suggestions.CodeGenerations", mock_suggestions_output) as mock:
        yield mock


@pytest.fixture
def mock_completions_stream(mock_suggestions_output: CodeSuggestionsOutput):
     with _mock_async_execute("ai_gateway.code_suggestions.CodeCompletions", mock_suggestions_output) as mock:
        yield mock


@pytest.fixture
def mock_with_prompt_prepared():
    with patch("ai_gateway.code_suggestions.CodeGenerations.with_prompt_prepared") as mock:
        yield mock


@pytest.fixture
def mock_litellm_acompletion():
    with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content="Test response"),
                    text="Test text completion response",
                ),
            ]
        )

        yield mock_acompletion


@pytest.fixture
def mock_litellm_acompletion_streamed():
    with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
        streamed_response = AsyncMock()
        streamed_response.__aiter__.return_value = iter(
            [
                AsyncMock(
                    choices=[AsyncMock(delta=AsyncMock(content="Streamed content"))]
                )
            ]
        )

        mock_acompletion.return_value = streamed_response

        yield mock_acompletion


@pytest.fixture
def model_response():
    yield "Hello there!"


@pytest.fixture
def model_engine():
    yield "fake-engine"


@pytest.fixture
def model_name():
    yield "fake-model"

@pytest.fixture
def model_error():
    yield None

class FakeModel(FakeListChatModel):
    model_engine: str
    model_name: str
    model_error: Optional[Exception] = None

    @property
    def _llm_type(self) -> str:
        return self.model_engine

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {**super()._identifying_params, **{"model": self.model_name}}
    
    async def _astream(
        self, *args, **kwargs,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for c in super(FakeModel, self)._astream(*args, **kwargs):
            yield c

        if self.model_error:
            raise self.model_error

@pytest.fixture
def model(model_response: str, model_engine: str, model_name: str, model_error: Exception):
    # our default Assistant prompt template already contains "Thought: "
    text = model_response.removeprefix("Thought: ") if model_response else ""

    yield FakeModel(model_engine=model_engine, model_name=model_name, responses=[text], model_error=model_error)


@pytest.fixture
def model_factory(model: BaseChatModel):
    yield lambda *args, **kwargs: model


@pytest.fixture
def model_params():
    yield ChatLiteLLMParams(model_class_provider="litellm")


@pytest.fixture
def model_config(model_params: TypeModelParams):
    yield ModelConfig(name="test_model", params=model_params)


@pytest.fixture
def prompt_template():
    yield {"system": "Hi, I'm {{name}}", "user": "{{content}}"}


@pytest.fixture
def unit_primitives():
    yield ["analyze_ci_job_failure"]


@pytest.fixture
def prompt_params():
    yield PromptParams()


@pytest.fixture
def prompt_config(
    model_config: ModelConfig,
    unit_primitives: list[GitLabUnitPrimitive],
    prompt_template: dict[str, str],
    prompt_params: PromptParams,
):
    yield PromptConfig(
        name="test_prompt",
        model=model_config,
        unit_primitives=unit_primitives,
        prompt_template=prompt_template,
        params=prompt_params,
    )


@pytest.fixture
def model_metadata():
    yield None

@pytest.fixture
def prompt_kwargs():
    yield {}

@pytest.fixture
def prompt_class():
    yield Prompt


@pytest.fixture
def prompt(
    prompt_class: Type[Prompt],
    model_factory: TypeModelFactory,
    prompt_config: PromptConfig,
    model_metadata: ModelMetadata | None,
    prompt_kwargs: dict,
):
    yield prompt_class(model_factory, prompt_config, model_metadata, **prompt_kwargs)
