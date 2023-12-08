from contextlib import contextmanager
from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from dependency_injector.providers import Factory

from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeSuggestionsChunk,
)
from ai_gateway.code_suggestions.processing import (
    LanguageId,
    ModelEngineCompletions,
    ModelEngineOutput,
)
from ai_gateway.code_suggestions.processing.post.completions import PostProcessor
from ai_gateway.code_suggestions.processing.pre import (
    PromptBuilderPrefixBased,
    TokenStrategyBase,
)
from ai_gateway.code_suggestions.processing.typing import (
    LanguageId,
    MetadataCodeContent,
    MetadataPromptBuilder,
)
from ai_gateway.instrumentators import KnownMetrics, TextGenModelInstrumentator
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    ModelAPIError,
    ModelMetadata,
    PalmCodeGeckoModel,
    SafetyAttributes,
    TextGenBaseModel,
    TextGenModelChunk,
    TextGenModelOutput,
)


class InstrumentorMock(Mock):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.watcher = Mock()

    @contextmanager
    def watch(self, _prompt: str, **_kwargs: Any):
        yield self.watcher


@pytest.mark.asyncio
class TestCodeCompletionsLegacy:
    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "engine_response_text",
            "expected_language_id",
            "expected_output",
        ),
        [
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                "a good suggestion",
                LanguageId.PYTHON,
                "a wonderful suggestion",
            ),
        ],
    )
    async def test_execute(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        engine_response_text: str,
        expected_language_id: LanguageId,
        expected_output: str,
    ):
        engine_response = ModelEngineOutput(
            text=engine_response_text,
            score=0,
            model=ModelMetadata(name="code-gecko@latest", engine="vertex-ai"),
            lang_id=expected_language_id,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                },
            ),
        )
        engine = Mock(spec=ModelEngineCompletions)
        engine.generate = AsyncMock(return_value=engine_response)
        engine.model = PalmCodeGeckoModel(
            client=Mock(), project="gl", location="us-central-1"
        )

        post_processor = Mock(spec=PostProcessor)
        post_processor.process.return_value = expected_output
        post_processor_factory = Mock()
        post_processor_factory.return_value = post_processor

        with patch(
            "ai_gateway.code_suggestions.completions.benchmark"
        ) as mock_benchmark:
            use_case = CodeCompletionsLegacy(
                engine=engine, post_processor=post_processor_factory
            )
            actual = await use_case.execute(
                prefix=prefix,
                suffix=suffix,
                file_name=file_name,
                editor_lang=editor_lang,
            )

        assert expected_output == actual.text
        assert expected_language_id == actual.lang_id

        engine.generate.assert_called_with(prefix, suffix, file_name, editor_lang)
        mock_benchmark.assert_called_with(
            metric_key=KnownMetrics.POST_PROCESSING_DURATION,
            labels={"model_engine": "vertex-ai", "model_name": "code-gecko@latest"},
        )
        post_processor_factory.assert_called_with(
            prefix, suffix=suffix, lang_id=expected_language_id
        )
        post_processor.process.assert_called_with(engine_response_text)

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "engine_response_text",
            "expected_language_id",
            "expected_output",
        ),
        [
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                "",
                LanguageId.PYTHON,
                "random_suggestion",
            ),
        ],
    )
    async def test_execute_without_post_processing(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        engine_response_text: str,
        expected_language_id: LanguageId,
        expected_output: str,
    ):
        engine_response = ModelEngineOutput(
            text=engine_response_text,
            score=0,
            model=ModelMetadata(name="code-gecko@latest", engine="vertex-ai"),
            lang_id=expected_language_id,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                },
            ),
        )
        engine = Mock(spec=ModelEngineCompletions)
        engine.generate = AsyncMock(return_value=engine_response)
        engine.model = PalmCodeGeckoModel(
            client=Mock(), project="gl", location="us-central-1"
        )

        post_processor = Mock(spec=PostProcessor)
        post_processor.process.return_value = expected_output
        post_processor_factory = Mock()
        post_processor_factory.return_value = post_processor

        with patch(
            "ai_gateway.code_suggestions.completions.benchmark"
        ) as mock_benchmark:
            use_case = CodeCompletionsLegacy(
                engine=engine, post_processor=post_processor_factory
            )
            _ = await use_case.execute(
                prefix=prefix,
                suffix=suffix,
                file_name=file_name,
                editor_lang=editor_lang,
            )

        mock_benchmark.assert_not_called()
        post_processor.process.assert_not_called()


@pytest.mark.asyncio
class TestCodeCompletions:
    @pytest.fixture(scope="class")
    def use_case(self):
        model = Mock(spec=TextGenBaseModel)
        model.MAX_MODEL_LEN = 2048

        use_case = CodeCompletions(model, Mock(spec=TokenStrategyBase))
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)
        use_case.prompt_builder = Mock(spec=PromptBuilderPrefixBased)

        yield use_case

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "stream",
            "expected_language_id",
            "expected_output",
        ),
        [
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                False,
                LanguageId.PYTHON,
                "random_suggestion",
            ),
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                None,
                False,
                None,
                "random_suggestion",
            ),
            (
                "random_prefix",
                "random_suffix",
                "file_name.py",
                None,
                False,
                LanguageId.PYTHON,
                "random_suggestion",
            ),
        ],
    )
    async def test_execute(
        self,
        use_case: CodeCompletions,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        stream: bool,
        expected_language_id: LanguageId,
        expected_output: str,
    ):
        use_case.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output, score=0, safety_attributes=SafetyAttributes()
            )
        )

        actual = await use_case.execute(
            prefix=prefix,
            suffix=suffix,
            file_name=file_name,
            editor_lang=editor_lang,
            stream=stream,
        )

        assert expected_output == actual.text
        assert expected_language_id == actual.lang_id

        use_case.model.generate.assert_called_with(
            use_case.prompt_builder.build().prefix,
            use_case.prompt_builder.build().suffix,
            stream,
        )

    @pytest.mark.parametrize(
        (
            "model_chunks",
            "expected_chunks",
        ),
        [
            (
                [
                    TextGenModelChunk(text="hello "),
                    TextGenModelChunk(text="world!"),
                ],
                [
                    "hello ",
                    "world!",
                ],
            ),
        ],
    )
    async def test_execute_stream(
        self,
        use_case: CodeCompletions,
        model_chunks: list[TextGenModelChunk],
        expected_chunks: list[str],
    ):
        async def _stream_generator(_prefix, _suffix, _stream):
            for chunk in model_chunks:
                yield chunk

        use_case.model.generate = AsyncMock(side_effect=_stream_generator)

        actual = await use_case.execute(
            prefix="any",
            suffix="how",
            file_name="bar.py",
            editor_lang=LanguageId.PYTHON,
            stream=True,
        )

        chunks = []
        async for content in actual:
            chunks += content

        assert chunks == expected_chunks

        use_case.model.generate.assert_called_with(
            use_case.prompt_builder.build().prefix,
            use_case.prompt_builder.build().suffix,
            True,
        )

    @pytest.mark.parametrize(
        (
            "prompt",
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "stream",
            "expected_language_id",
            "expected_output",
        ),
        [
            (
                "prompt_random_prefix",
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                False,
                LanguageId.PYTHON,
                "random_suggestion",
            ),
        ],
    )
    async def test_execute_with_prompt_prepared(
        self,
        use_case: CodeCompletions,
        prompt: str,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        stream: bool,
        expected_language_id: LanguageId,
        expected_output: str,
    ):
        use_case.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output, score=0, safety_attributes=SafetyAttributes()
            )
        )

        actual = await use_case.execute(
            prefix, suffix, file_name, editor_lang=editor_lang, raw_prompt=prompt
        )

        assert expected_output == actual.text
        assert expected_language_id == actual.lang_id

        use_case.model.generate.assert_called_with(
            use_case.prompt_builder.wrap().prefix,
            use_case.prompt_builder.wrap().suffix,
            stream,
        )

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "model_exception_type",
        ),
        [
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                AnthropicAPIStatusError,
            ),
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                AnthropicAPIConnectionError,
            ),
        ],
    )
    async def test_execute_caught_exception(
        self,
        use_case: CodeCompletions,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        model_exception_type: Type[ModelAPIError],
    ):
        if issubclass(model_exception_type, AnthropicAPIStatusError):
            model_exception_type.code = 404
        exception = model_exception_type("exception message")

        def _side_effect(*_args, **_kwargs):
            raise exception

        use_case.model.generate = AsyncMock(side_effect=_side_effect)

        _ = await use_case.execute(prefix, suffix, file_name, editor_lang)

        code = (
            model_exception_type.code if hasattr(model_exception_type, "code") else -1
        )

        use_case.instrumentator.watcher.register_model_exception.assert_called_with(
            str(exception), code
        )
