"""Composable retrieval strategies, each returning a ranked list of event uids.

Only chunking variants need a fresh index; everything here happens at query time on a
shared one, which is what keeps the ablation cheap to extend.
"""

from collections.abc import Callable
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


class HybridSearch:
    """Reciprocal rank fusion of a dense and a lexical strategy.

    RRF needs no score calibration between the two, which is what makes it safe to
    combine a cosine similarity with a BM25 score.
    """

    def __init__(
        self,
        dense: SearchStrategy,
        lexical: SearchStrategy,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[str]:
        pool = k * OVERFETCH
        scores: dict[str, float] = {}
        for strategy in (self._dense, self._lexical):
            for rank, uid in enumerate(strategy.search(query, pool), start=1):
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (self._rrf_k + rank)
        ranked = sorted(scores, key=lambda uid: scores[uid], reverse=True)
        return ranked[:k]


class CityFilter:
    """Keep only events from the town a query names.

    Town names are matched against the set present in the corpus — a lookup, not language
    understanding. When a query names no known town the filter passes results through
    untouched, so it never removes what it cannot justify removing.
    """

    def __init__(self, inner: SearchStrategy, city_by_uid: dict[str, str]) -> None:
        self._inner = inner
        self._city_by_uid = city_by_uid
        # Longest first, so "Castelnaudary" wins over "Castelnau".
        self._cities = sorted({c for c in city_by_uid.values() if c}, key=len, reverse=True)

    def _named_city(self, query: str) -> str | None:
        lowered = query.lower()
        for city in self._cities:
            if city.lower() in lowered:
                return city
        return None

    def search(self, query: str, k: int) -> list[str]:
        city = self._named_city(query)
        if city is None:
            return self._inner.search(query, k)
        candidates = self._inner.search(query, k * OVERFETCH)
        return [uid for uid in candidates if self._city_by_uid.get(uid) == city][:k]


class Rerank:
    """Re-order an inner strategy's candidates with a scoring function.

    The scorer is injected rather than constructed here, so the unit tests need no model
    and swapping the reranking backend touches one call site.
    """

    def __init__(
        self,
        inner: SearchStrategy,
        scorer: Callable[[str, list[str]], list[float]],
        candidates_factor: int = 4,
    ) -> None:
        self._inner = inner
        self._scorer = scorer
        self._factor = candidates_factor

    def search(self, query: str, k: int) -> list[str]:
        candidates = self._inner.search(query, k * self._factor)
        if not candidates:
            return []
        scores = self._scorer(query, candidates)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order][:k]
