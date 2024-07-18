from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.base import LANGUAGE_COUNTER
from ai_gateway.code_suggestions.processing.ops import (
    lang_from_editor_lang,
    lang_from_filename,
)
from ai_gateway.experimentation import ExperimentTelemetry
from ai_gateway.models import (
    KindAnthropicModel,
    KindLiteLlmModel,
    KindModelProvider,
    KindVertexTextModel,
    ModelMetadata,
)

__all__ = [
    "KindUseCase",
    "CodeSuggestionsOutput",
    "CodeSuggestionsChunk",
    "ModelProvider",
    "PROVIDERS_MODELS_MAP",
    "USE_CASES_MODELS_MAP",
]


class ModelProvider(str, Enum):
    VERTEX_AI = "vertex-ai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"


class KindUseCase(str, Enum):
    CODE_COMPLETIONS = "code completions"
    CODE_GENERATIONS = "code generations"


PROVIDERS_MODELS_MAP = {
    KindModelProvider.ANTHROPIC: KindAnthropicModel,
    KindModelProvider.VERTEX_AI: KindVertexTextModel,
    KindModelProvider.LITELLM: KindLiteLlmModel,
    KindModelProvider.MISTRALAI: KindLiteLlmModel,
}

USE_CASES_MODELS_MAP = {
    KindUseCase.CODE_COMPLETIONS: {
        KindAnthropicModel.CLAUDE_INSTANT_1_1,
        KindAnthropicModel.CLAUDE_INSTANT_1_2,
        KindVertexTextModel.CODE_GECKO_002,
        KindLiteLlmModel.MISTRAL,
        KindLiteLlmModel.MIXTRAL,
        KindLiteLlmModel.CODE_GEMMA,
        KindLiteLlmModel.CODESTRAL,
        KindLiteLlmModel.CODE_LLAMA_CODE,
    },
    KindUseCase.CODE_GENERATIONS: {
        KindAnthropicModel.CLAUDE_2_0,
        KindAnthropicModel.CLAUDE_2_1,
        KindVertexTextModel.CODE_BISON_002,
        KindAnthropicModel.CLAUDE_3_OPUS,
        KindAnthropicModel.CLAUDE_3_SONNET,
        KindAnthropicModel.CLAUDE_3_5_SONNET,
        KindAnthropicModel.CLAUDE_3_HAIKU,
        KindLiteLlmModel.MISTRAL,
        KindLiteLlmModel.MIXTRAL,
        KindLiteLlmModel.CODE_GEMMA,
        KindLiteLlmModel.CODESTRAL,
        KindLiteLlmModel.CODE_LLAMA,
    },
}


class CodeSuggestionsOutput(NamedTuple):
    class Metadata(NamedTuple):
        experiments: list[ExperimentTelemetry]

    text: str
    score: float
    model: ModelMetadata
    lang_id: Optional[LanguageId] = None
    metadata: Optional[Metadata] = None

    @property
    def lang(self) -> str:
        return self.lang_id.name.lower() if self.lang_id else ""


class CodeSuggestionsChunk(NamedTuple):
    text: str


def resolve_lang_id(
    file_name: str, editor_lang: Optional[str] = None
) -> Optional[LanguageId]:
    lang_id = lang_from_filename(file_name)

    if lang_id is None and editor_lang:
        lang_id = lang_from_editor_lang(editor_lang)

    return lang_id


# TODO: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/292
def increment_lang_counter(
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
