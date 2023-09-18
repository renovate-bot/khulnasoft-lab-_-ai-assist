from typing import Any, Callable, NamedTuple, Optional

import structlog
from dependency_injector.providers import Factory
from transformers import PreTrainedTokenizer

from ai_gateway.code_suggestions.processing.base import (
    MINIMIMUM_CONFIDENCE_SCORE,
    ModelEngineBase,
    ModelEngineOutput,
    Prompt,
    PromptBuilderBase,
)
from ai_gateway.code_suggestions.processing.ops import truncate_content
from ai_gateway.code_suggestions.processing.post.completions import PostProcessor
from ai_gateway.code_suggestions.processing.typing import (
    CodeContent,
    LanguageId,
    MetadataCodeContent,
    MetadataExtraInfo,
    MetadataPromptBuilder,
)
from ai_gateway.experimentation import ExperimentRegistry, ExperimentTelemetry
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import (
    PalmCodeGenBaseModel,
    VertexModelInternalError,
    VertexModelInvalidArgument,
)
from ai_gateway.prompts.parsers import CodeParser

log = structlog.stdlib.get_logger("codesuggestions")

__all__ = [
    "ModelEngineCompletions",
]

_KEY_EXAMPLE_LANG_ID = {
    "python": LanguageId.PYTHON,
}


class _CodeBody(NamedTuple):
    prefix: CodeContent
    suffix: CodeContent


class _CodeInfo(NamedTuple):
    content: list[CodeContent]

    @property
    def total_length_tokens(self):
        return sum(info.length_tokens for info in self.content)

    @property
    def total_length(self):
        return sum(len(info.text) for info in self.content)


def _double_slash_comment(comment: str) -> str:
    return f"// {comment}"


# TODO: Convert these to templates later
COMMENT_GENERATOR: dict[LanguageId, Callable[[str], str]] = {
    LanguageId.C: lambda comment: f"/* {comment} */",
    LanguageId.CPP: _double_slash_comment,
    LanguageId.CSHARP: _double_slash_comment,
    LanguageId.GO: _double_slash_comment,
    LanguageId.JAVA: _double_slash_comment,
    LanguageId.JS: _double_slash_comment,
    LanguageId.PHP: _double_slash_comment,
    LanguageId.PYTHON: lambda comment: f"# {comment}",
    LanguageId.RUBY: lambda comment: f"# {comment}",
    LanguageId.RUST: _double_slash_comment,
    LanguageId.SCALA: _double_slash_comment,
    LanguageId.TS: _double_slash_comment,
    LanguageId.KOTLIN: _double_slash_comment,
}


class _PromptBuilder(PromptBuilderBase):
    LANG_ID_TO_HUMAN_NAME = {
        LanguageId.C: "C",
        LanguageId.CPP: "C++",
        LanguageId.CSHARP: "C#",
        LanguageId.GO: "Go",
        LanguageId.JAVA: "Java",
        LanguageId.JS: "JavaScript",
        LanguageId.PHP: "PHP",
        LanguageId.PYTHON: "Python",
        LanguageId.RUBY: "Ruby",
        LanguageId.RUST: "Rust",
        LanguageId.SCALA: "Scala",
        LanguageId.TS: "TypeScript",
        LanguageId.KOTLIN: "Kotlin",
    }

    def __init__(
        self,
        prefix: CodeContent,
        suffix: CodeContent,
        file_name: str,
        lang_id: Optional[LanguageId] = None,
        experiments: Optional[list[ExperimentTelemetry]] = [],
    ):
        super().__init__(prefix, suffix, lang_id)

        self.file_name = file_name
        self._metadata["experiments"] = experiments

    def add_extra_info(
        self, extra_info: _CodeInfo, max_total_length_tokens: int, extra_info_name: str
    ):
        total_length_tokens = 0
        total_length = 0

        # Only prepend the info if it's not present and we have room
        for info in extra_info.content:
            if info.text in self._prefix or info.text in self._suffix:
                continue

            total_length += len(info.text)
            total_length_tokens += info.length_tokens
            if max_total_length_tokens - total_length_tokens >= 0:
                self._prefix = f"{info.text}\n{self._prefix}"

        self._metadata[extra_info_name] = MetadataExtraInfo(
            name=extra_info_name,
            pre=MetadataCodeContent(
                length=extra_info.total_length,
                length_tokens=extra_info.total_length_tokens,
            ),
            post=MetadataCodeContent(
                length=total_length,
                length_tokens=total_length_tokens,
            ),
        )

    def _prepend_comments(self) -> str:
        if self.lang_id not in COMMENT_GENERATOR:
            header = f"This code has a filename of {self.file_name}"
            return f"{header}\n{self._prefix}"

        comment = COMMENT_GENERATOR[self.lang_id]
        language = self.LANG_ID_TO_HUMAN_NAME[self.lang_id]
        header = comment(
            f"This code has a filename of {self.file_name} and is written in {language}."
        )
        return f"{header}\n{self._prefix}"

    def build(self) -> Prompt:
        new_prefix = self._prepend_comments()

        return Prompt(
            prefix=new_prefix,
            suffix=self._suffix,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": self._metadata["prefix"],
                    "suffix": self._metadata["suffix"],
                },
                imports=self._metadata.get("imports", None),
                function_signatures=self._metadata.get("function_signatures", None),
                experiments=self._metadata["experiments"],
            ),
        )


class ModelEngineCompletions(ModelEngineBase):
    MAX_TOKENS_IMPORTS_PERCENT = 0.12  # about 245 tokens for code-gecko
    MAX_TOKENS_SUFFIX_PERCENT = 0.07  # about 126 tokens for code-gecko, if "imports" takes up all the available space

    def __init__(
        self,
        model: PalmCodeGenBaseModel,
        tokenizer: PreTrainedTokenizer,
        post_processor: Factory[PostProcessor],
        experiment_registry: ExperimentRegistry,
    ):
        super().__init__(model, tokenizer)
        self.post_processor_factory = post_processor
        self.experiment_registry = experiment_registry

    async def _generate(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        lang_id: LanguageId,
        **kwargs: Any,
    ) -> ModelEngineOutput:
        prompt = self._build_prompt(prefix, file_name, suffix, lang_id)

        empty_output = ModelEngineOutput(
            text="",
            score=0,
            model=self.model.metadata,
            metadata=MetadataPromptBuilder(components={}),
        )

        # TODO: keep watching the suffix length until logging ModelEngineOutput in the upper layer
        with self.instrumentator.watch(
            prompt, suffix_length=len(suffix)
        ) as watch_container:
            try:
                # count symbols of the final prompt
                self._count_symbols(prompt.prefix, lang_id, watch_container)

                # log experiments included in this request
                self._count_experiments(prompt.metadata.experiments, watch_container)

                if res := await self.model.generate(
                    prompt.prefix, prompt.suffix, **kwargs
                ):
                    watch_container.register_model_output_length(res.text)
                    watch_container.register_model_score(res.score)

                    if res.score > MINIMIMUM_CONFIDENCE_SCORE:
                        # TODO: Move the call to the use case class
                        completion = self.post_processor_factory(
                            prompt.prefix, lang_id=lang_id
                        ).process(res.text)
                    else:
                        watch_container.register_is_discarded()
                        completion = ""

                    return ModelEngineOutput(
                        text=completion,
                        score=res.score,
                        model=self.model.metadata,
                        lang_id=lang_id,
                        metadata=prompt.metadata,
                    )
            except (VertexModelInvalidArgument, VertexModelInternalError) as ex:
                watch_container.register_model_exception(str(ex), ex.code)

        return empty_output

    def _build_prompt(
        self,
        prefix: str,
        file_name: str,
        suffix: str,
        lang_id: Optional[LanguageId] = None,
    ) -> Prompt:
        imports = self._get_imports(prefix, lang_id)
        prompt_len_imports_max = int(
            self.model.MAX_MODEL_LEN * self.MAX_TOKENS_IMPORTS_PERCENT
        )
        prompt_len_imports = min(imports.total_length_tokens, prompt_len_imports_max)

        func_signatures = self._get_function_signatures(suffix, lang_id)
        prompt_len_func_signatures = min(
            func_signatures.total_length_tokens, 1024
        )  # max 1024 tokens

        prompt_len_body = (
            self.model.MAX_MODEL_LEN - prompt_len_imports - prompt_len_func_signatures
        )

        experiments = []
        if exp := self.experiment_registry.get_experiment("exp_truncate_suffix"):
            experiment_output = exp.run(
                logger=log, prefix=prefix, suffix=suffix, lang_id=lang_id
            )
            experiments.append(experiment_output.telemetry)
            truncated_suffix = experiment_output.output
            body = self._get_body(prefix, truncated_suffix, prompt_len_body)
        else:
            body = self._get_body(prefix, suffix, prompt_len_body)

        prompt_builder = _PromptBuilder(
            body.prefix, body.suffix, file_name, lang_id, experiments
        )
        # NOTE that the last thing we add here will appear first in the prefix
        prompt_builder.add_extra_info(
            func_signatures,
            prompt_len_func_signatures,
            extra_info_name="function_signatures",
        )
        prompt_builder.add_extra_info(
            imports, prompt_len_imports, extra_info_name="imports"
        )
        prompt = prompt_builder.build()

        return prompt

    def _get_imports(
        self, content: str, lang_id: Optional[LanguageId] = None
    ) -> _CodeInfo:
        imports = self._extract(content, "imports", lang_id)
        return self._to_code_info(imports, lang_id, as_comments=False)

    def _get_function_signatures(
        self, content: str, lang_id: Optional[LanguageId] = None
    ) -> _CodeInfo:
        signatures = self._extract(content, "function_signatures", lang_id)
        return self._to_code_info(signatures, lang_id, as_comments=True)

    @staticmethod
    def _extract(
        content: str, target: str, lang_id: Optional[LanguageId] = None
    ) -> list[str]:
        extracted = []
        if lang_id:
            try:
                parser = CodeParser.from_language_id(content, lang_id)
                if target == "imports":
                    extracted = parser.imports()
                elif target == "function_signatures":
                    extracted = parser.function_signatures()
                else:
                    raise ValueError(f"Unknown extraction target {target}")
            except ValueError as e:
                log.warning(f"Failed to parse code: {e}")

        return extracted

    def _to_code_info(
        self, contents: list[str], lang_id: LanguageId, as_comments: bool = True
    ) -> _CodeInfo:
        """
        Convert a list of code snippets into `_CodeInfo`, which includes metadata like text length and token length.
        """
        if len(contents) == 0:
            return _CodeInfo(content=[])

        if as_comments:
            comment_converter = COMMENT_GENERATOR[lang_id]
            contents = [comment_converter(content) for content in contents]

        contents_tokenized = self.tokenizer(
            contents,
            return_length=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        code_contents = [
            CodeContent(text=text, length_tokens=length)
            for text, length in zip(contents, contents_tokenized["length"])
        ]

        return _CodeInfo(content=code_contents)

    def _get_body(self, prefix: str, suffix: str, max_length: int) -> _CodeBody:
        suffix_len = int(max_length * self.MAX_TOKENS_SUFFIX_PERCENT)
        suffix_truncated = truncate_content(
            self.tokenizer,
            suffix,
            max_length=suffix_len,
            truncation_side="right",
        )

        prefix_len = max_length - suffix_truncated.length_tokens
        prefix_truncated = truncate_content(
            self.tokenizer,
            prefix,
            max_length=prefix_len,
            truncation_side="left",
        )

        return _CodeBody(prefix=prefix_truncated, suffix=suffix_truncated)

    def _count_symbols(
        self,
        prompt: str,
        lang_id: LanguageId,
        watch_container: TextGenModelInstrumentator.WatchContainer,
    ) -> None:
        try:
            parser = CodeParser.from_language_id(prompt, lang_id)
            symbol_map = parser.count_symbols()
            self.increment_code_symbol_counter(lang_id, symbol_map)
            self.log_symbol_map(watch_container, symbol_map)
        except ValueError as e:
            log.warning(f"Failed to parse code: {e}")

    def _count_experiments(
        self,
        experiments: list[ExperimentTelemetry],
        watch_container: TextGenModelInstrumentator.WatchContainer,
    ) -> None:
        watch_container.register_experiments(experiments)
        self.increment_experiment_counter(experiments)
