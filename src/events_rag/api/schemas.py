"""The request and response shapes, which are also the generated OpenAPI documentation.

`AskResponse` carries the sources beside the answer on purpose: an answer whose supporting
events cannot be checked is the failure mode this whole repository is arranged against.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceItem(BaseModel):
    uid: str
    title: str
    city: str | None = None
    date: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class MetadataResponse(BaseModel):
    location_field: str
    location_value: str
    language: str
    date_window_mode: str
    date_window_days: int
    retrieval_k: int


class RebuildRequest(BaseModel):
    token: str = Field(min_length=1)


class RebuildResponse(BaseModel):
    status: str
    indexed_documents: int
    indexed_chunks: int
    index_path: str
