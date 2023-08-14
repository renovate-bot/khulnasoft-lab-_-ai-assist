from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional

from prometheus_client import Counter

from codesuggestions.instrumentators import TextGenModelInstrumentator
from codesuggestions.suggestions.processing.ops import LanguageId, lang_from_filename

__all__ = [
    "MetadataCodeContent",
    "MetadataExtraInfo",
    "MetadataPromptBuilder",
    "ModelEngineOutput",
    "MetadataModel",
    "ModelEngineBase",
]

LANGUAGE_COUNTER = Counter(
    "code_suggestions_prompt_language",
    "Language count by number",
    ["lang", "extension"],
)

CODE_SYMBOL_COUNTER = Counter(
    "code_suggestions_prompt_symbols", "Prompt symbols count", ["lang", "symbol"]
)


class MetadataCodeContent(NamedTuple):
    length: int
    length_tokens: int


class MetadataExtraInfo(NamedTuple):
    name: str
    pre: MetadataCodeContent
    post: MetadataCodeContent


class MetadataPromptBuilder(NamedTuple):
    components: Mapping[str, MetadataCodeContent]
    imports: Optional[MetadataExtraInfo] = None
    function_signatures: Optional[MetadataExtraInfo] = None


class MetadataModel(NamedTuple):
    name: str
    engine: str


class ModelEngineOutput(NamedTuple):
    text: str
    model: MetadataModel
    lang_id: Optional[LanguageId] = None
    metadata: Optional[MetadataPromptBuilder] = None

    def lang(self) -> str:
        return self.lang_id.name.lower() if self.lang_id else ""


class ModelEngineBase(ABC):
    async def generate(
        self, prefix: str, suffix: str, file_name: str, **kwargs: Any
    ) -> ModelEngineOutput:
        lang_id = lang_from_filename(file_name)
        self.increment_lang_counter(file_name, lang_id)
        return await self._generate(prefix, suffix, file_name, lang_id, **kwargs)

    def increment_lang_counter(
        self, filename: str, lang_id: Optional[LanguageId] = None
    ):
        labels = {"lang": None}

        if lang_id:
            labels["lang"] = lang_id.name.lower()

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
