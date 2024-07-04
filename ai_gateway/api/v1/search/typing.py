from typing import Annotated, List

from pydantic import BaseModel, Field, StringConstraints

__all__ = [
    "SearchRequest",
    "SearchResponseMetadata",
    "SearchResponse",
    "SearchResponseDetails",
]

DEFAULT_PAGE_SIZE = 4


class SearchMetadata(BaseModel):
    source: Annotated[str, StringConstraints(max_length=100)]
    version: Annotated[str, StringConstraints(max_length=100)]


class SearchPayload(BaseModel):
    query: Annotated[str, StringConstraints(max_length=400000)]
    page_size: Annotated[int, Field(ge=1, le=20)] = DEFAULT_PAGE_SIZE


class SearchRequest(BaseModel):
    type: Annotated[str, StringConstraints(max_length=100)]
    metadata: SearchMetadata
    payload: SearchPayload


class SearchResponseMetadata(BaseModel):
    provider: str
    timestamp: int


class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict


class SearchResponseDetails(BaseModel):
    results: List[SearchResult]


class SearchResponse(BaseModel):
    response: SearchResponseDetails
    metadata: SearchResponseMetadata
