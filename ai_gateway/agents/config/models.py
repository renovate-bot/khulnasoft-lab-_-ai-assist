from enum import Enum
from typing import Annotated, Literal, Mapping

from pydantic import BaseModel, Field

__all__ = [
    "ModelClassProvider",
    "TypeModelParams",
    "BaseModelParams",
    "ChatLiteLLMParams",
    "ChatAnthropicParams",
]


class ModelClassProvider(str, Enum):
    LITE_LLM = "litellm"
    ANTHROPIC = "anthropic"


class BaseModelParams(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_retries: int | None = 1


class ChatLiteLLMParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.LITE_LLM]
    custom_llm_provider: str | None = None
    """Easily switch to huggingface, replicate, together ai, sagemaker, etc.
    Example - https://litellm.vercel.app/docs/providers/vllm#batch-completion"""


class ChatAnthropicParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.ANTHROPIC]
    default_headers: Mapping[str, str] | None = None


TypeModelParams = Annotated[
    ChatLiteLLMParams | ChatAnthropicParams, Field(discriminator="model_class_provider")
]
