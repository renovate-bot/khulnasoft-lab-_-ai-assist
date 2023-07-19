from pathlib import Path
from typing import Optional, Any, NamedTuple
import structlog

from transformers import PreTrainedTokenizer

from codesuggestions.models import TextGenBaseModel
from codesuggestions.prompts import PromptTemplateBase, PromptTemplate, PromptTemplateFewShot
from codesuggestions.prompts.code_parser import CodeParser
from codesuggestions.suggestions.processing.base import LanguageId, ModelEngineBase
from codesuggestions.suggestions.processing.ops import (
    trim_by_max_len,
    trim_by_sep,
    lang_from_filename,
    prepend_lang_id,
    remove_incomplete_lines,
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

    @property
    def total_length_tokens(self):
        return self.prefix.length_tokens + self.suffix.length_tokens


class _CodeImports(NamedTuple):
    content: list[_CodeContent]

    @property
    def total_length_tokens(self):
        return sum([import_statement.length_tokens for import_statement in self.content])


class _PromptBuilder:
    def __init__(self, prefix: str, suffix: str):
        self._prefix = prefix
        self._suffix = suffix

    def add_imports(self, imports: _CodeImports, max_total_length_tokens: int):
        total_length_tokens = 0

        # Only prepend the import statement if it's not present and we have room
        for import_statement in imports.content:
            if (
                self._prefix.find(import_statement.text) >= 0
                or self._suffix.find(import_statement.text) >= 0
            ):
                continue

            total_length_tokens += import_statement.length_tokens
            if max_total_length_tokens - total_length_tokens >= 0:
                self._prefix = f"{import_statement.text}\n{self._prefix}"

    def build(self) -> tuple[str, str]:
        return self._prefix, self._suffix


class ModelEngineCodegen(ModelEngineBase):
    FILE_EXAMPLES = "examples.json"
    EXAMPLES_TEMPLATE = "base.tpl"
    COMPLETION_TEMPLATE = "completion.tpl"

    SEP_CODE_BLOCK = "```"

    def __init__(
        self,
        model: TextGenBaseModel,
        prompt_tpls: dict[LanguageId, PromptTemplateBase],
        sep_code_block: str
    ):
        self.model = model
        self.prompt_tpls = prompt_tpls
        self.sep_code_block = sep_code_block

    def generate_completion(self, prefix: str, suffix: str, file_name: str, **kwargs: Any) -> str:
        # collect metrics
        lang_id = lang_from_filename(file_name)
        self.increment_lang_counter(file_name, lang_id)

        prompt = self._build_prompt(prefix, lang_id)
        if res := self.model.generate(prompt, suffix, **kwargs):
            completion = self._clean_completions(res.text)
            return completion

        return ""

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
            tpl = PromptTemplateFewShot.from_local_file(tpl_dir / tpl_completion, examples, tpl_examples)
            prompt_tpls[_KEY_EXAMPLE_LANG_ID[key_example]] = tpl

        return cls(model, prompt_tpls, sep_code_block)


class ModelEnginePalm(ModelEngineBase):
    # TODO: implement another custom prompt template here
    def __init__(self, model: TextGenBaseModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_completion(self, prefix: str, suffix: str, file_name: str, **kwargs: Any):
        # collect metrics
        lang_id = lang_from_filename(file_name)
        self.increment_lang_counter(file_name, lang_id)

        prompt, suffix = self._build_prompt(prefix, suffix, lang_id)

        # count symbols of the final prompt
        self._count_symbols(prompt, lang_id)

        if res := self.model.generate(prompt, suffix, **kwargs):
            return res.text

        return ""

    def _build_prompt(self, prefix: str, suffix: str, lang_id: Optional[LanguageId]) -> tuple[str, str]:
        imports = self._get_imports(prefix, lang_id)
        prompt_len_imports = min(imports.total_length_tokens, 512)  # max 512 tokens
        prompt_len_body = self.model.MAX_MODEL_LEN - prompt_len_imports

        body = self._get_body(prefix, suffix, prompt_len_body)

        prompt_builder = _PromptBuilder(body.prefix.text, body.suffix.text)
        prompt_builder.add_imports(imports, prompt_len_imports)
        prefix, suffix = prompt_builder.build()

        return prefix, suffix

    def _get_imports(self, content: str, lang_id: Optional[LanguageId] = None) -> _CodeImports:
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
            for import_text, length in zip(imports_extracted, imports_tokenized["length"])
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

    def _truncate_content(self, val: str, max_length: int, truncation_side: str = "left") -> _CodeContent:
        self.tokenizer.truncation_side = truncation_side

        tokens = self.tokenizer(
            val,
            max_length=max_length,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        decoded = self.tokenizer.decode(tokens['input_ids'])

        return _CodeContent(
            text=decoded,
            length_tokens=len(tokens["input_ids"]),
        )

    def _count_symbols(self, prompt: str, lang_id: LanguageId) -> None:
        try:
            symbol_map = CodeParser(lang_id).count_symbols(prompt, target_symbols={"imports"})
            self.increment_code_symbol_counter(lang_id, symbol_map)
        except ValueError as e:
            log.warning(f"Failed to parse code: {e}")
