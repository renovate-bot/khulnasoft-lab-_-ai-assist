from enum import Enum
from typing import Annotated, Literal, Mapping, Optional

from pydantic import BaseModel, Field

__all__ = [
    "ModelClassProvider",
    "TypeModelParams",
    "BaseModelParams",
    "ChatLiteLLMParams",
    "ChatAnthropicParams",
]


class ModelClassProvider(str, Enum):
    LITE_LLM = "lite_llm"
    ANTHROPIC = "anthropic"


class BaseModelParams(BaseModel):
    temperature: Optional[float] = 1.0
    timeout: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: Optional[int] = 2_048
    max_retries: Optional[int] = 1


class ChatLiteLLMParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.LITE_LLM]
    timeout: Annotated[float | None, Field(serialization_alias="request_timeout")] = (
        None
    )
    custom_llm_provider: str | None = None
    """Easily switch to huggingface, replicate, together ai, sagemaker, etc.
    Example - https://litellm.vercel.app/docs/providers/vllm#batch-completion"""


class ChatAnthropicParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.ANTHROPIC]
    timeout: Annotated[
        float | None, Field(serialization_alias="default_request_timeout")
    ] = None
    default_headers: Optional[Mapping[str, str]] = None


TypeModelParams = Annotated[
    ChatLiteLLMParams | ChatAnthropicParams, Field(discriminator="model_class_provider")
]
