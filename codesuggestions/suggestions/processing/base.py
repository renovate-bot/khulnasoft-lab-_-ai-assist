from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple, Optional

from prometheus_client import Counter
from transformers import PreTrainedTokenizer

from codesuggestions.experimentation import ExperimentTelemetry
from codesuggestions.instrumentators import TextGenModelInstrumentator
from codesuggestions.models import PalmCodeGenBaseModel
from codesuggestions.suggestions.processing.ops import lang_from_filename
from codesuggestions.suggestions.processing.typing import (
    CodeContent,
    LanguageId,
    MetadataCodeContent,
    MetadataModel,
    MetadataPromptBuilder,
)

__all__ = [
    "ModelEngineOutput",
    "ModelEngineBase",
    "Prompt",
    "PromptBuilderBase",
]

LANGUAGE_COUNTER = Counter(
    "code_suggestions_prompt_language",
    "Language count by number",
    ["lang", "extension", "editor_lang"],
)

CODE_SYMBOL_COUNTER = Counter(
    "code_suggestions_prompt_symbols", "Prompt symbols count", ["lang", "symbol"]
)

EXPERIMENT_COUNTER = Counter(
    "code_suggestions_experiments", "Ongoing experiments", ["name", "variant"]
)


class ModelEngineOutput(NamedTuple):
    text: str
    model: MetadataModel
    metadata: MetadataPromptBuilder
    lang_id: Optional[LanguageId] = None

    def lang(self) -> str:
        return self.lang_id.name.lower() if self.lang_id else ""


class ModelEngineBase(ABC):
    def __init__(self, model: PalmCodeGenBaseModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.instrumentator = TextGenModelInstrumentator(
            model.model_engine, model.model_name
        )

    async def generate(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang_id: Optional[str] = None,
        **kwargs: Any
    ) -> ModelEngineOutput:
        lang_id = lang_from_filename(file_name)
        self.increment_lang_counter(file_name, lang_id, editor_lang_id)
        return await self._generate(prefix, suffix, file_name, lang_id, **kwargs)

    def increment_lang_counter(
        self,
        filename: str,
        lang_id: Optional[LanguageId] = None,
        editor_lang_id: Optional[str] = None,
    ):
        labels = {"lang": None, "editor_lang": None}

        if lang_id:
            labels["lang"] = lang_id.name.lower()

        if editor_lang_id:
            labels["editor_lang"] = editor_lang_id

        labels["extension"] = Path(filename).suffix[1:]

        LANGUAGE_COUNTER.labels(**labels).inc()

    @abstractmethod
    async def _generate(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        lang_id: LanguageId,
        **kwargs: Any
    ) -> ModelEngineOutput:
        pass

    def increment_code_symbol_counter(self, lang_id: LanguageId, symbol_map: dict):
        for symbol, count in symbol_map.items():
            CODE_SYMBOL_COUNTER.labels(lang=lang_id.name.lower(), symbol=symbol).inc(
                count
            )

    def log_symbol_map(
        self,
        watch_container: TextGenModelInstrumentator.WatchContainer,
        symbol_map: dict,
    ) -> None:
        watch_container.register_prompt_symbols(symbol_map)

    def increment_experiment_counter(self, experiments: list[ExperimentTelemetry]):
        for exp in experiments:
            EXPERIMENT_COUNTER.labels(name=exp.name, variant=exp.variant).inc()


class Prompt(NamedTuple):
    prefix: str
    metadata: MetadataPromptBuilder
    suffix: Optional[str] = None


class PromptBuilderBase(ABC):
    def __init__(
        self,
        prefix: CodeContent,
        suffix: Optional[CodeContent] = None,
        lang_id: Optional[LanguageId] = None,
    ):
        self.lang_id = lang_id
        self._prefix = prefix.text

        self._metadata = {
            "prefix": MetadataCodeContent(
                length=len(prefix.text),
                length_tokens=prefix.length_tokens,
            ),
        }

        if suffix:
            self._suffix = suffix.text
            self._metadata["suffix"] = MetadataCodeContent(
                length=len(suffix.text),
                length_tokens=suffix.length_tokens,
            )

    @abstractmethod
    def build(self) -> Prompt:
        pass
