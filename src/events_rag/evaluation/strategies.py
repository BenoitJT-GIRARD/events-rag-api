"""Composable retrieval strategies, each returning a ranked list of event uids.

Only chunking variants need a fresh index; everything here happens at query time on a
shared one, which is what keeps the ablation cheap to extend.
"""

from typing import Protocol

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from events_rag.evaluation.retrieval import dedupe_uids

# One event is split into several chunks, so a top-k of chunks collapses to fewer
# distinct events. Over-fetch before deduplicating, or a strategy returns short.
OVERFETCH = 4


class SearchStrategy(Protocol):
    def search(self, query: str, k: int) -> list[str]:
        """Return at most k distinct event uids, best first."""


def _uids(documents: list[Document]) -> list[str]:
    return [str(doc.metadata.get("uid", "")) for doc in documents if doc.metadata.get("uid")]


class DenseSearch:
    def __init__(self, vectorstore) -> None:
        self._vectorstore = vectorstore

    def search(self, query: str, k: int) -> list[str]:
        documents = self._vectorstore.similarity_search(query, k=k * OVERFETCH)
        return dedupe_uids(_uids(documents))[:k]


class BM25Search:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._index = BM25Okapi([doc.page_content.lower().split() for doc in documents])

    def search(self, query: str, k: int) -> list[str]:
        scores = self._index.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked = [self._documents[i] for i in order[: k * OVERFETCH]]
        return dedupe_uids(_uids(ranked))[:k]
