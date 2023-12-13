from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from fastapi import Body
from pydantic import BaseModel, Field, constr

from ai_gateway.code_suggestions import ModelProvider

__all__ = [
    "CompletionRequest",
]


class ComponentType(str, Enum):
    CODE_EDITOR_COMPLETION = "code_editor_completion"
    CODE_EDITOR_GENERATION = "code_editor_generation"


class MetadataBase(BaseModel):
    source: constr(max_length=255)
    version: constr(max_length=255)


class EditorContentPayload(BaseModel):
    file_name: constr(strip_whitespace=True, max_length=255)
    content_above_cursor: constr(max_length=100000)
    content_below_cursor: constr(max_length=100000)
    language_identifier: Optional[constr(max_length=255)]
    model_provider: Optional[Literal[ModelProvider.VERTEX_AI, ModelProvider.ANTHROPIC]]
    stream: Optional[bool] = False


class EditorContentCompletionPayload(EditorContentPayload):
    pass


class EditorContentGenerationPayload(EditorContentPayload):
    prompt: Optional[constr(max_length=400000)]


class CodeEditorCompletion(BaseModel):
    type: Literal[ComponentType.CODE_EDITOR_COMPLETION]
    payload: EditorContentCompletionPayload
    metadata: Optional[MetadataBase]


class CodeEditorGeneration(BaseModel):
    type: Literal[ComponentType.CODE_EDITOR_GENERATION]
    payload: EditorContentGenerationPayload
    metadata: Optional[MetadataBase]


PromptComponent = Annotated[
    Union[CodeEditorCompletion, CodeEditorGeneration], Body(discriminator="type")
]


class CompletionRequest(BaseModel):
    prompt_components: Annotated[List[PromptComponent], Field(min_items=1, max_items=1)]


class ModelMetadata(BaseModel):
    engine: Optional[str]
    name: Optional[str]
    lang: Optional[str]


class ResponseMetadataBase(BaseModel):
    model: Optional[ModelMetadata]
    timestamp: int


class CompletionResponse(BaseModel):
    response: Optional[str]
    metadata: Optional[ResponseMetadataBase]
