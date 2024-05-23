from typing import Annotated, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
)
from starlette.responses import StreamingResponse

from ai_gateway.code_suggestions import (
    PROVIDERS_MODELS_MAP,
    USE_CASES_MODELS_MAP,
    KindUseCase,
)
from ai_gateway.experimentation import ExperimentTelemetry
from ai_gateway.instrumentators.base import Telemetry
from ai_gateway.models import KindModelProvider, Message

__all__ = [
    "CompletionsRequestV1",
    "GenerationsRequestV1",
    "CompletionsRequestV2",
    "GenerationsRequestV2",
    "SuggestionsResponse",
    "StreamSuggestionsResponse",
]


class CurrentFile(BaseModel):
    file_name: Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]
    language_identifier: Optional[Annotated[str, StringConstraints(max_length=255)]] = (
        None  # https://code.visualstudio.com/docs/languages/identifiers
    )
    content_above_cursor: Annotated[str, StringConstraints(max_length=100000)]
    content_below_cursor: Annotated[str, StringConstraints(max_length=100000)]


class CodeContextPayload(BaseModel):
    type: Annotated[str, StringConstraints(max_length=1024)]
    name: Annotated[str, StringConstraints(max_length=1024)]
    content: Annotated[str, StringConstraints(max_length=100000)]


class SuggestionsRequest(BaseModel):
    # Opt out protected namespace "model_" (https://github.com/pydantic/pydantic/issues/6322).
    model_config = ConfigDict(protected_namespaces=())

    project_path: Optional[
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]
    ] = None
    project_id: Optional[int] = None
    current_file: CurrentFile
    model_provider: Optional[KindModelProvider] = None
    model_endpoint: Optional[str] = None
    model_name: Optional[
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=50)]
    ] = None

    telemetry: Annotated[List[Telemetry], Field(max_length=10)] = []
    stream: Optional[bool] = False
    choices_count: Optional[int] = 0
    context: Annotated[List[CodeContextPayload], Field(max_length=100)] = []


class CompletionsRequest(SuggestionsRequest):
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str, info: ValidationInfo) -> str:
        """Validate model name and model provider are compatible."""

        return _validate_model_name(
            value, KindUseCase.CODE_COMPLETIONS, info.data.get("model_provider")
        )


class GenerationsRequest(SuggestionsRequest):
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str, info: ValidationInfo) -> str:
        """Validate model name and model provider are compatible."""

        return _validate_model_name(
            value, KindUseCase.CODE_GENERATIONS, info.data.get("model_provider")
        )


class CompletionsRequestV1(CompletionsRequest):
    prompt_version: Literal[1] = 1


class GenerationsRequestV1(GenerationsRequest):
    prompt_version: Literal[1] = 1


class CompletionsRequestV2(CompletionsRequest):
    prompt_version: Literal[2]
    prompt: str


class GenerationsRequestV2(GenerationsRequest):
    prompt_version: Literal[2]
    prompt: str


class GenerationsRequestV3(GenerationsRequest):
    prompt_version: Literal[3]
    prompt: list[Message]


class SuggestionsResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    class Model(BaseModel):
        engine: str
        name: str
        lang: str

    id: str
    model: Model
    experiments: list[ExperimentTelemetry] = []
    object: str = "text_completion"
    created: int
    choices: list[Choice]


class StreamSuggestionsResponse(StreamingResponse):
    pass


def _validate_model_name(
    model_name: str,
    use_case: KindUseCase,
    provider: Optional[KindModelProvider] = None,
) -> str:
    # ignore model name validation when provider is invalid
    if not provider:
        return model_name

    use_case_models = USE_CASES_MODELS_MAP.get(use_case)
    provider_models = PROVIDERS_MODELS_MAP.get(provider)

    if not use_case_models or not provider_models:
        raise ValueError(f"model {model_name} is unknown")

    valid_model_names = use_case_models & set(provider_models)

    if model_name not in valid_model_names:
        raise ValueError(
            f"model {model_name} is not supported by use case {use_case} and provider {provider}"
        )

    return model_name
