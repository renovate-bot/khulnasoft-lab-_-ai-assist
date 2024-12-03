from enum import StrEnum
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
)

from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sse_starlette.sse import EventSourceResponse
from starlette.responses import StreamingResponse

from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeGenerations,
    CodeSuggestionsChunk,
    ModelProvider,
)
from ai_gateway.models import KindVertexTextModel, Message

__all__ = [
    "CodeEditorComponents",
    "CompletionRequest",
    "CompletionResponse",
    "EditorContentCompletionPayload",
    "EditorContentGenerationPayload",
    "StreamSuggestionsResponse",
]


class CodeEditorComponents(StrEnum):
    COMPLETION = "code_editor_completion"
    GENERATION = "code_editor_generation"
    CONTEXT = "code_context"


class MetadataBase(BaseModel):
    source: Annotated[str, StringConstraints(max_length=255)]
    version: Annotated[str, StringConstraints(max_length=255)]


class EditorContentPayload(BaseModel):
    # Opt out protected namespace "model_" (https://github.com/pydantic/pydantic/issues/6322).
    model_config = ConfigDict(protected_namespaces=())

    file_name: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=255)
    ] = Field(examples=["example.py"])
    content_above_cursor: Annotated[str, StringConstraints(max_length=100000)] = Field(
        examples=["def hello_world():\n    print("]
    )
    content_below_cursor: Annotated[str, StringConstraints(max_length=100000)] = Field(
        examples=[""]
    )
    language_identifier: Optional[Annotated[str, StringConstraints(max_length=255)]] = (
        Field(None, examples=["python"])
    )
    model_provider: Optional[
        Literal[ModelProvider.VERTEX_AI, ModelProvider.ANTHROPIC]
    ] = None
    stream: Optional[bool] = False


class EditorContentCompletionPayload(EditorContentPayload):
    choices_count: Optional[int] = 0
    model_name: Optional[str] = Field(
        None, examples=[KindVertexTextModel.CODE_GECKO_002]
    )
    prompt: Optional[str | list[Message]] = Field(
        None, examples=["Complete the function"]
    )


class EditorContentGenerationPayload(EditorContentPayload):
    prompt: Optional[Annotated[str, StringConstraints(max_length=400000)]] = None
    prompt_id: Optional[str] = None
    prompt_enhancer: Optional[dict[str, Any]] = None


class CodeEditorCompletion(BaseModel):
    type: Literal[CodeEditorComponents.COMPLETION]
    payload: EditorContentCompletionPayload
    metadata: Optional[MetadataBase] = None


class CodeEditorGeneration(BaseModel):
    type: Literal[CodeEditorComponents.GENERATION]
    payload: EditorContentGenerationPayload
    metadata: Optional[MetadataBase] = None


class CodeContextPayload(BaseModel):
    type: Annotated[str, StringConstraints(max_length=1024)]
    name: Annotated[str, StringConstraints(max_length=1024)]
    content: Annotated[str, StringConstraints(max_length=100000)]


class CodeContext(BaseModel):
    type: Literal[CodeEditorComponents.CONTEXT]
    payload: CodeContextPayload
    metadata: Optional[MetadataBase] = None


PromptComponent = Annotated[
    Union[CodeEditorCompletion, CodeEditorGeneration, CodeContext],
    Body(discriminator="type"),
]


class CompletionRequest(BaseModel):
    prompt_components: Annotated[
        List[PromptComponent], Field(min_length=1, max_length=100)
    ]


class ModelMetadata(BaseModel):
    engine: Optional[str] = None
    name: Optional[str] = None
    lang: Optional[str] = None


class ResponseMetadataBase(BaseModel):
    model: Optional[ModelMetadata] = None
    timestamp: int


class CompletionResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    choices: list[Choice]
    metadata: Optional[ResponseMetadataBase] = None


class StreamSuggestionsResponse(StreamingResponse):
    pass


# Only includes engines that support streaming
StreamModelEngine = Union[CodeCompletions, CodeGenerations]


class StreamHandler(Protocol):
    async def __call__(
        self,
        stream: AsyncIterator[CodeSuggestionsChunk],
        engine: StreamModelEngine,
    ) -> Union[StreamSuggestionsResponse, EventSourceResponse]:
        pass
