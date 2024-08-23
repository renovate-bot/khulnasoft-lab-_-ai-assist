from typing import Annotated, Any, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import AnyUrl, BaseModel, StringConstraints, UrlConstraints, validator

STUBBED_API_KEY = "<api-key>"


class ModelMetadata(BaseModel):
    name: Annotated[str, StringConstraints(max_length=100)]
    provider: Annotated[str, StringConstraints(max_length=100)]
    endpoint: Annotated[AnyUrl, UrlConstraints(max_length=100)]
    api_key: Optional[Annotated[str, StringConstraints(max_length=100)]] = None

    # OpenAI client requires api key to be set
    @validator("api_key", pre=True, always=True)
    @classmethod
    def set_stubbed_api_key_if_empty(cls, v):
        return v or STUBBED_API_KEY


class TypeModelFactory(Protocol):
    def __call__(self, *, model: str, **kwargs: Optional[Any]) -> BaseChatModel: ...
