from typing import Annotated, Any, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import AnyUrl, BaseModel, StringConstraints, UrlConstraints


class ModelMetadata(BaseModel):
    name: Annotated[str, StringConstraints(max_length=255)]
    provider: Annotated[str, StringConstraints(max_length=255)]
    endpoint: Annotated[AnyUrl, UrlConstraints(max_length=255)]
    api_key: Optional[Annotated[str, StringConstraints(max_length=255)]] = None
    identifier: Optional[Annotated[str, StringConstraints(max_length=255)]] = None


class TypeModelFactory(Protocol):
    def __call__(self, *, model: str, **kwargs: Optional[Any]) -> BaseChatModel: ...
