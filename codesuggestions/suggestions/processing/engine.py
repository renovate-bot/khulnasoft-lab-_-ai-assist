from pathlib import Path
from typing import Optional, Any, NamedTuple

from transformers import PreTrainedTokenizer

from codesuggestions.models import TextGenBaseModel
from codesuggestions.prompts import PromptTemplateBase, PromptTemplate, PromptTemplateFewShot
from codesuggestions.prompts.import_extractor import ImportExtractor
from codesuggestions.suggestions.processing.base import LanguageId, ModelEngineBase
from codesuggestions.suggestions.processing.ops import (
    trim_by_max_len,
    trim_by_sep,
    lang_from_filename,
    prepend_lang_id,
    remove_incomplete_lines,
)

__all__ = [
    "ModelEngineCodegen",
    "ModelEnginePalm",
]

_KEY_EXAMPLE_LANG_ID = {
    "python": LanguageId.PYTHON,
}


class _ContentTruncated(NamedTuple):
    val: str
    length_tokens: int


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
        if res := self.model.generate(prompt, suffix, **kwargs):
            return res.text

        return ""

    def _build_prompt(self, prefix: str, suffix: str, lang_id: Optional[LanguageId]) -> tuple[str, str]:
        suffix_truncated = self._truncate_content(
            suffix,
            max_length=self.model.MAX_MODEL_LEN // 2,
            truncation_side="right",
        )
        prefix_truncated = self._truncate_content(
            prefix,
            max_length=self.model.MAX_MODEL_LEN - suffix_truncated.length_tokens,
            truncation_side="left",
        )

        # TODO: Once the tokenizer is implemented, we either take `MAX_MODEL_LEN` tokens
        # TODO: and there is no more room for imports or the prompt already contains the import statements
        # TODO: if the prefix is less than 2048 tokens.
        # Wait for https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/174
        # to get more room to include imports
        # prompt = self._add_imports(prefix, prompt, lang_id)

        return prefix_truncated.val, suffix_truncated.val

    def _add_imports(self, content: str, prompt: str, lang_id: Optional[LanguageId]) -> str:
        """
        Deprecated.
        """
        extractor = ImportExtractor(lang_id)
        imports = extractor.extract_imports(content)

        if imports is None:
            return prompt

        for text in imports:
            # Only prepend the import statement if it's not present and we have room.
            # The model truncates excess tokens that precede the cursor
            if prompt.find(text) == -1 and (len(prompt) + len(text) < self.model.UPPER_BOUND_MODEL_CHARS):
                prompt = text + prompt + "\n"

        return prompt

    def _truncate_content(self, val: str, max_length: int, truncation_side: str = "left") -> _ContentTruncated:
        self.tokenizer.truncation_side = truncation_side

        tokens = self.tokenizer(
            val,
            max_length=max_length,
            truncation=True,
            return_length=True,
            return_attention_mask=False,
        )

        decoded = self.tokenizer.decode(
            tokens['input_ids'],
            skip_special_tokens=True,
        )

        return _ContentTruncated(
            val=decoded,
            length_tokens=tokens['length'],
        )
