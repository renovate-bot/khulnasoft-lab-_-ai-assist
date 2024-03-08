from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple, Optional

from prometheus_client import Counter

from ai_gateway.code_suggestions.processing.ops import (
    lang_from_editor_lang,
    lang_from_filename,
)
from ai_gateway.code_suggestions.processing.typing import (
    CodeContent,
    LanguageId,
    MetadataCodeContent,
    MetadataPromptBuilder,
    Prompt,
    TokenStrategyBase,
)
from ai_gateway.experimentation import ExperimentTelemetry
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import ModelMetadata, PalmCodeGenBaseModel
from ai_gateway.models.base import TokensConsumptionMetadata

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

MINIMIMUM_CONFIDENCE_SCORE = -10


class ModelEngineOutput(NamedTuple):
    text: str
    score: float
    model: ModelMetadata
    metadata: MetadataPromptBuilder
    tokens_consumption_metadata: TokensConsumptionMetadata
    lang_id: Optional[LanguageId] = None

    @property
    def lang(self) -> str:
        return self.lang_id.name.lower() if self.lang_id else ""


class ModelEngineBase(ABC):
    def __init__(
        self, model: PalmCodeGenBaseModel, tokenization_strategy: TokenStrategyBase
    ):
        self.model = model
        self.tokenization_strategy = tokenization_strategy
        self.instrumentator = TextGenModelInstrumentator(
            model.metadata.engine, model.metadata.name
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

        if lang_id is None and editor_lang_id:
            lang_id = lang_from_editor_lang(editor_lang_id)

        return await self._generate(
            prefix, suffix, file_name, lang_id, editor_lang_id, **kwargs
        )

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
        editor_lang: Optional[str] = None,
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
