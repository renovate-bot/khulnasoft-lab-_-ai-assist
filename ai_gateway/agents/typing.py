from typing import Annotated, Optional

from pydantic import AnyUrl, BaseModel, StringConstraints, UrlConstraints


class ModelMetadata(BaseModel):
    endpoint: Annotated[AnyUrl, UrlConstraints(max_length=100)]
    name: Annotated[str, StringConstraints(max_length=100)]
    provider: Annotated[str, StringConstraints(max_length=100)]
    api_key: Optional[Annotated[str, StringConstraints(max_length=100)]] = None
