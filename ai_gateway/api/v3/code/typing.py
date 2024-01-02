from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.responses import StreamingResponse

from ai_gateway.code_suggestions import ModelProvider

__all__ = [
    "CodeEditorComponents",
    "CompletionRequest",
    "CompletionResponse",
    "EditorContentCompletionPayload",
    "EditorContentGenerationPayload",
    "StreamSuggestionsResponse",
]


class CodeEditorComponents(str, Enum):
    COMPLETION = "code_editor_completion"
    GENERATION = "code_editor_generation"


class MetadataBase(BaseModel):
    source: Annotated[str, StringConstraints(max_length=255)]
    version: Annotated[str, StringConstraints(max_length=255)]


class EditorContentPayload(BaseModel):
    # Opt out protected namespace "model_" (https://github.com/pydantic/pydantic/issues/6322).
    model_config = ConfigDict(protected_namespaces=())

    file_name: Annotated[str, StringConstraints(strip_whitespace=True, max_length=255)]
    content_above_cursor: Annotated[str, StringConstraints(max_length=100000)]
    content_below_cursor: Annotated[str, StringConstraints(max_length=100000)]
    language_identifier: Optional[
        Annotated[str, StringConstraints(max_length=255)]
    ] = None
    model_provider: Optional[
        Literal[ModelProvider.VERTEX_AI, ModelProvider.ANTHROPIC]
    ] = None
    stream: Optional[bool] = False


class EditorContentCompletionPayload(EditorContentPayload):
    pass


class EditorContentGenerationPayload(EditorContentPayload):
    prompt: Optional[Annotated[str, StringConstraints(max_length=400000)]] = None


class CodeEditorCompletion(BaseModel):
    type: Literal[CodeEditorComponents.COMPLETION]
    payload: EditorContentCompletionPayload
    metadata: Optional[MetadataBase] = None


class CodeEditorGeneration(BaseModel):
    type: Literal[CodeEditorComponents.GENERATION]
    payload: EditorContentGenerationPayload
    metadata: Optional[MetadataBase] = None


PromptComponent = Annotated[
    Union[CodeEditorCompletion, CodeEditorGeneration], Body(discriminator="type")
]


class CompletionRequest(BaseModel):
    prompt_components: Annotated[
        List[PromptComponent], Field(min_length=1, max_length=1)
    ]


class ModelMetadata(BaseModel):
    engine: Optional[str] = None
    name: Optional[str] = None
    lang: Optional[str] = None


class ResponseMetadataBase(BaseModel):
    model: Optional[ModelMetadata] = None
    timestamp: int


class CompletionResponse(BaseModel):
    response: Optional[str] = None
    metadata: Optional[ResponseMetadataBase] = None


class StreamSuggestionsResponse(StreamingResponse):
    pass
