from pathlib import Path
from typing import Any, NamedTuple, Optional

import structlog
from transformers import PreTrainedTokenizer

from codesuggestions.instrumentators import TextGenModelInstrumentator
from codesuggestions.models import (
    PalmCodeGenBaseModel,
    TextGenBaseModel,
    VertexModelInternalError,
    VertexModelInvalidArgument,
)
from codesuggestions.prompts import (
    PromptTemplate,
    PromptTemplateBase,
    PromptTemplateFewShot,
)
from codesuggestions.prompts.code_parser import CodeParser
from codesuggestions.suggestions.processing.base import (
    MetadataCodeContent,
    MetadataImports,
    MetadataModel,
    MetadataPromptBuilder,
    ModelEngineBase,
    ModelEngineOutput,
)
from codesuggestions.suggestions.processing.ops import (
    LanguageId,
    prepend_lang_id,
    remove_incomplete_lines,
    trim_by_max_len,
    trim_by_sep,
)

log = structlog.stdlib.get_logger("codesuggestions")

__all__ = [
    "ModelEngineCodegen",
    "ModelEnginePalm",
]

_KEY_EXAMPLE_LANG_ID = {
    "python": LanguageId.PYTHON,
}


class _CodeContent(NamedTuple):
    text: str
    length_tokens: int


class _CodeBody(NamedTuple):
    prefix: _CodeContent
    suffix: _CodeContent


class _CodeImports(NamedTuple):
    content: list[_CodeContent]

    @property
    def total_length_tokens(self):
        return sum(
            [import_statement.length_tokens for import_statement in self.content]
        )

    @property
    def total_length(self):
        return sum([len(import_statement.text) for import_statement in self.content])


class _Prompt(NamedTuple):
    prefix: str
    suffix: str
    metadata: MetadataPromptBuilder


def _double_slash_comment(comment):
    return f"// {comment}"


class _PromptBuilder:
    DOUBLE_SLASH_COMMENT = _double_slash_comment

    # TODO: Convert these to templates later
    COMMENT_GENERATOR = {
        LanguageId.C: lambda comment: f"/* {comment} */",
        LanguageId.CPP: DOUBLE_SLASH_COMMENT,
        LanguageId.CSHARP: DOUBLE_SLASH_COMMENT,
        LanguageId.GO: DOUBLE_SLASH_COMMENT,
        LanguageId.JAVA: DOUBLE_SLASH_COMMENT,
        LanguageId.JS: DOUBLE_SLASH_COMMENT,
        LanguageId.PHP: DOUBLE_SLASH_COMMENT,
        LanguageId.PYTHON: lambda comment: f"# {comment}",
        LanguageId.RUBY: lambda comment: f"# {comment}",
        LanguageId.RUST: DOUBLE_SLASH_COMMENT,
        LanguageId.SCALA: DOUBLE_SLASH_COMMENT,
        LanguageId.TS: DOUBLE_SLASH_COMMENT,
        LanguageId.KOTLIN: DOUBLE_SLASH_COMMENT,
    }
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
        prefix: _CodeContent,
        suffix: _CodeContent,
        file_name: str,
        lang_id: Optional[LanguageId] = None,
    ):
        self.lang_id = lang_id
        self.file_name = file_name

        self._prefix = prefix.text
        self._suffix = suffix.text

        self._metadata = {
            "prefix": MetadataCodeContent(
                length=len(prefix.text),
                length_tokens=prefix.length_tokens,
            ),
            "suffix": MetadataCodeContent(
                length=len(suffix.text),
                length_tokens=suffix.length_tokens,
            ),
        }

    def add_imports(self, imports: _CodeImports, max_total_length_tokens: int):
        total_length_tokens = 0
        total_length = 0

        # Only prepend the import statement if it's not present and we have room
        for import_statement in imports.content:
            if (
                self._prefix.find(import_statement.text) >= 0
                or self._suffix.find(import_statement.text) >= 0
            ):
                continue

            total_length += len(import_statement.text)
            total_length_tokens += import_statement.length_tokens
            if max_total_length_tokens - total_length_tokens >= 0:
                self._prefix = f"{import_statement.text}\n{self._prefix}"

        self._metadata["imports"] = MetadataImports(
            pre=MetadataCodeContent(
                length=imports.total_length,
                length_tokens=imports.total_length_tokens,
            ),
            post=MetadataCodeContent(
                length=total_length,
                length_tokens=total_length_tokens,
            ),
        )

    def _prepend_comments(self) -> str:
        if self.lang_id not in self.COMMENT_GENERATOR:
            header = f"This code has a filename of {self.file_name}"
            return f"{header}\n{self._prefix}"

        comment = self.COMMENT_GENERATOR[self.lang_id]
        language = self.LANG_ID_TO_HUMAN_NAME[self.lang_id]
        header = comment(
            f"This code has a filename of {self.file_name} and is written in {language}."
        )
        return f"{header}\n{self._prefix}"

    def build(self) -> _Prompt:
        new_prefix = self._prepend_comments()

        return _Prompt(
            prefix=new_prefix,
            suffix=self._suffix,
            metadata=MetadataPromptBuilder(
                prefix=self._metadata["prefix"],
                suffix=self._metadata["suffix"],
                imports=self._metadata.get("imports", None),
            ),
        )


class ModelEngineCodegen(ModelEngineBase):
    FILE_EXAMPLES = "examples.json"
    EXAMPLES_TEMPLATE = "base.tpl"
    COMPLETION_TEMPLATE = "completion.tpl"

    SEP_CODE_BLOCK = "```"

    def __init__(
        self,
        model: TextGenBaseModel,
        prompt_tpls: dict[LanguageId, PromptTemplateBase],
        sep_code_block: str,
    ):
        self.model = model
        self.prompt_tpls = prompt_tpls
        self.sep_code_block = sep_code_block

    async def _generate_completion(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        lang_id: LanguageId,
        **kwargs: Any,
    ) -> ModelEngineOutput:
        prompt = self._build_prompt(prefix, lang_id)
        model_metadata = MetadataModel(
            name=self.model.model_name, engine=self.model.model_engine
        )

        if res := self.model.generate(prompt, suffix, **kwargs):
            completion = self._clean_completions(res.text)
            return ModelEngineOutput(
                text=completion,
                model=model_metadata,
            )

        return ModelEngineOutput(text="", model=model_metadata)

    def _build_prompt(self, content: str, lang_id: Optional[LanguageId]) -> str:
        prompt = trim_by_max_len(content, self.model.MAX_MODEL_LEN)

        if prompt_tpl := self.prompt_tpls.get(lang_id, None):
            prompt = prompt_tpl.apply(lang=lang_id.name.lower(), prompt=prompt)
        else:
            prompt = prepend_lang_id(prompt, lang_id)

        return prompt

    def _clean_completions(self, completion: str) -> str:
        completion = remove_incomplete_lines(
            trim_by_sep(completion, sep=self.sep_code_block)
        )

        return completion

    @classmethod
    def from_local_templates(
        cls,
        tpl_dir: Path,
        model: TextGenBaseModel,
        tpl_completion: str = COMPLETION_TEMPLATE,
        tpl_examples: str = EXAMPLES_TEMPLATE,
        file_examples: str = FILE_EXAMPLES,
        sep_code_block: str = SEP_CODE_BLOCK,
    ):
        all_examples = cls._read_json(tpl_dir / file_examples)

        prompt_tpls = dict()
        for key_example, examples in all_examples.items():
            tpl_examples = PromptTemplate.from_local_file(tpl_dir / tpl_examples)
            tpl = PromptTemplateFewShot.from_local_file(
                tpl_dir / tpl_completion, examples, tpl_examples
            )
            prompt_tpls[_KEY_EXAMPLE_LANG_ID[key_example]] = tpl

        return cls(model, prompt_tpls, sep_code_block)


class ModelEnginePalm(ModelEngineBase):
    def __init__(self, model: PalmCodeGenBaseModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.instrumentator = TextGenModelInstrumentator(
            model.model_engine, model.model_name
        )

    async def _generate_completion(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        lang_id: LanguageId,
        **kwargs: Any,
    ) -> ModelEngineOutput:
        prompt = self._build_prompt(prefix, file_name, suffix, lang_id)

        model_metadata = MetadataModel(
            name=self.model.model_name, engine=self.model.model_engine
        )
        empty_output = ModelEngineOutput(text="", model=model_metadata)

        # TODO: keep watching the suffix length until logging ModelEngineOutput in the upper layer
        with self.instrumentator.watch(
            prompt.prefix, suffix_length=len(suffix)
        ) as watch_container:
            try:
                # count symbols of the final prompt
                self._count_symbols(prompt.prefix, lang_id, watch_container)

                if res := await self.model.generate(
                    prompt.prefix, prompt.suffix, **kwargs
                ):
                    return ModelEngineOutput(
                        text=res.text,
                        model=model_metadata,
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
    ) -> _Prompt:
        imports = self._get_imports(prefix, lang_id)
        prompt_len_imports = min(imports.total_length_tokens, 512)  # max 512 tokens
        prompt_len_body = self.model.MAX_MODEL_LEN - prompt_len_imports

        body = self._get_body(prefix, suffix, prompt_len_body)

        prompt_builder = _PromptBuilder(body.prefix, body.suffix, file_name, lang_id)
        prompt_builder.add_imports(imports, prompt_len_imports)
        prompt = prompt_builder.build()

        return prompt

    def _get_imports(
        self, content: str, lang_id: Optional[LanguageId] = None
    ) -> _CodeImports:
        imports_extracted = []
        if lang_id:
            try:
                parser = CodeParser(lang_id)
                imports_extracted = parser.extract_imports(content)
            except ValueError as e:
                log.warning(f"Failed to parse code: {e}")

        if len(imports_extracted) == 0:
            return _CodeImports(content=[])

        imports_tokenized = self.tokenizer(
            imports_extracted,
            return_length=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        imports = [
            _CodeContent(text=import_text, length_tokens=length)
            for import_text, length in zip(
                imports_extracted, imports_tokenized["length"]
            )
        ]

        return _CodeImports(content=imports)

    def _get_body(self, prefix: str, suffix: str, max_length: int) -> _CodeBody:
        suffix_truncated = self._truncate_content(
            suffix,
            max_length=max_length // 2,
            truncation_side="right",
        )
        prefix_truncated = self._truncate_content(
            prefix,
            max_length=max_length - suffix_truncated.length_tokens,
            truncation_side="left",
        )

        return _CodeBody(prefix=prefix_truncated, suffix=suffix_truncated)

    def _truncate_content(
        self, val: str, max_length: int, truncation_side: str = "left"
    ) -> _CodeContent:
        self.tokenizer.truncation_side = truncation_side

        tokens = self.tokenizer(
            val,
            max_length=max_length,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        decoded = self.tokenizer.decode(tokens["input_ids"])

        return _CodeContent(
            text=decoded,
            length_tokens=len(tokens["input_ids"]),
        )

    def _count_symbols(
        self,
        prompt: str,
        lang_id: LanguageId,
        watch_container: TextGenModelInstrumentator.WatchContainer,
    ) -> None:
        try:
            symbol_map = CodeParser(lang_id).count_symbols(
                prompt, target_symbols=CodeParser.LANGUAGES_TARGETS.keys()
            )
            self.increment_code_symbol_counter(lang_id, symbol_map)
            self.log_symbol_map(watch_container, symbol_map)
        except ValueError as e:
            log.warning(f"Failed to parse code: {e}")
