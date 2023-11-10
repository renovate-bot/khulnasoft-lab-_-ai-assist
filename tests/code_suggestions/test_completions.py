from contextlib import contextmanager
from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from ai_gateway.code_suggestions import CodeCompletions, CodeSuggestionsChunk
from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.pre import (
    PromptBuilderPrefixBased,
    TokenStrategyBase,
)
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    ModelAPIError,
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
