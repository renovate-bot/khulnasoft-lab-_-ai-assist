from pathlib import Path
from typing import Optional, Any

from codesuggestions.models import TextGenBaseModel
from codesuggestions.prompts import PromptTemplateBase, PromptTemplate, PromptTemplateFewShot
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
    def __init__(self, model: TextGenBaseModel):
        self.model = model

    def generate_completion(self, prefix: str, suffix: str, file_name: str, **kwargs: Any):
        # collect metrics
        lang_id = lang_from_filename(file_name)
        self.increment_lang_counter(file_name, lang_id)

        prompt, suffix = self._build_prompt(prefix, suffix)
        if res := self.model.generate(prompt, suffix, **kwargs):
            return res.text

        return ""

    def _build_prompt(self, prefix: str, suffix: str) -> tuple[str, str]:
        suffix_len = min(len(suffix), self.model.MAX_MODEL_LEN // 2)
        prompt_len = self.model.MAX_MODEL_LEN - suffix_len

        prompt = trim_by_max_len(prefix, prompt_len)

        return prompt, suffix[:suffix_len]
