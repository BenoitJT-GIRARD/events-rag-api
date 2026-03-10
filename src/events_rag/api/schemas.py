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
    geography: str
    language: str
    retrieval_k: int