from enum import Enum
from typing import Mapping, NamedTuple, Optional

from ai_gateway.experimentation.base import ExperimentTelemetry

__all__ = [
    "LanguageId",
    "MetadataCodeContent",
    "MetadataExtraInfo",
    "MetadataPromptBuilder",
    "MetadataModel",
    "CodeContent",
]


class LanguageId(Enum):
    C = 1
    CPP = 2
    CSHARP = 3
    GO = 4
    JAVA = 5
    JS = 6
    PHP = 7
    PYTHON = 8
    RUBY = 9
    RUST = 10
    SCALA = 11
    TS = 12
    KOTLIN = 13


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
    experiments: Optional[list[ExperimentTelemetry]] = []


class MetadataModel(NamedTuple):
    name: str
    engine: str


class CodeContent(NamedTuple):
    text: str
    length_tokens: int
