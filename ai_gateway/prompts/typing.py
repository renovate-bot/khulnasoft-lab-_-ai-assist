from typing import Annotated, Any, Optional, Protocol, TypeAlias

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableBinding
from pydantic import AnyUrl, BaseModel, StringConstraints, UrlConstraints

# NOTE: Do not change this to `BaseChatModel | RunnableBinding`. You'd think that's just equivalent, right? WRONG. If
# you do that, you'll get `object has no attribute 'get'` when you use a `RummableBinding`. Why? I have no idea.
# https://docs.python.org/3/library/stdtypes.html#types-union makes no mention of the order mattering. This might be
# a bug with Pydantic's type validations
Model: TypeAlias = RunnableBinding | BaseChatModel


class ModelMetadata(BaseModel):
    name: Annotated[str, StringConstraints(max_length=255)]
    provider: Annotated[str, StringConstraints(max_length=255)]
    endpoint: Annotated[AnyUrl, UrlConstraints(max_length=255)]
    api_key: Optional[Annotated[str, StringConstraints(max_length=255)]] = None
    identifier: Optional[Annotated[str, StringConstraints(max_length=255)]] = None


class TypeModelFactory(Protocol):
    def __call__(self, *, model: str, **kwargs: Optional[Any]) -> Model: ...
