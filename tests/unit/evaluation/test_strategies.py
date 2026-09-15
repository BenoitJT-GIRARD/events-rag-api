"""Each retrieval strategy does what its name says.

Dense deduplicates by event and overfetches so that deduplication does not starve the
result, BM25 ranks the lexically closest document first, the hybrid fusion promotes what
both lists found without dropping what only one did, and the city filter keeps the town
named in the query — preferring the longest matching name, so that a query naming a town
whose name contains another one is not filtered to the wrong place.
"""

from langchain_core.documents import Document

from events_rag.evaluation.strategies import (
    BM25Search,
    CityFilter,
    DenseSearch,
    HybridSearch,
    Rerank,
)


class FakeVectorstore:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._documents[:k]


def _doc(uid: str, text: str) -> Document:
    return Document(page_content=text, metadata={"uid": uid})


def test_dense_search_returns_deduplicated_uids() -> None:
    store = FakeVectorstore([_doc("a", "one"), _doc("a", "two"), _doc("b", "three")])

    assert DenseSearch(store).search("anything", k=3) == ["a", "b"]


def test_dense_search_overfetches_so_dedup_does_not_starve_the_result() -> None:
    # Three chunks of "a" then one of "b": asking for k=2 must still surface "b".
    store = FakeVectorstore([_doc("a", "1"), _doc("a", "2"), _doc("a", "3"), _doc("b", "4")])

    assert DenseSearch(store).search("anything", k=2) == ["a", "b"]


def test_bm25_ranks_the_lexically_closest_document_first() -> None:
    # Five documents, not two: with a corpus of two, a term present in a single document
    # gets an IDF of log((2-1+0.5)/(1+0.5)) = 0, every score ties, and the ranking falls
    # back to insertion order. BM25 needs a corpus to discriminate anything.
    documents = [
        _doc("a", "exposition de peinture au musee"),
        _doc("b", "concert de jazz en plein air"),
        _doc("c", "visite guidee du chateau"),
        _doc("d", "atelier de danse pour enfants"),
        _doc("e", "marche de producteurs locaux"),
    ]

    assert BM25Search(documents).search("concert jazz", k=1) == ["b"]


class FakeStrategy:
    def __init__(self, uids: list[str]) -> None:
        self._uids = uids

    def search(self, query: str, k: int) -> list[str]:
        return self._uids[:k]


def test_hybrid_promotes_the_uid_both_lists_found() -> None:
    dense = FakeStrategy(["x", "b"])
    lexical = FakeStrategy(["y", "b"])

    # "b" is second in both lists and scores 2/62, while "x" and "y" are first in one
    # list only and score 1/61 each. Agreement across both sides is what RRF rewards.
    #
    # Note it does not reward closeness to the top: 1/61 + 1/63 exceeds 2/62, so being
    # first and third beats being second twice. Convexity of 1/x, not a bug.
    assert HybridSearch(dense, lexical).search("q", k=1) == ["b"]


def test_hybrid_keeps_a_uid_that_only_one_side_found() -> None:
    dense = FakeStrategy(["a"])
    lexical = FakeStrategy(["z"])

    assert set(HybridSearch(dense, lexical).search("q", k=2)) == {"a", "z"}


def test_city_filter_keeps_only_the_town_named_in_the_query() -> None:
    inner = FakeStrategy(["a", "b", "c"])
    cities = {"a": "Toulouse", "b": "Sete", "c": "Sete"}

    assert CityFilter(inner, cities).search("concerts a Sete ce soir", k=2) == ["b", "c"]


def test_city_filter_passes_through_when_no_known_town_is_named() -> None:
    inner = FakeStrategy(["a", "b"])
    cities = {"a": "Toulouse", "b": "Sete"}

    assert CityFilter(inner, cities).search("des concerts gratuits", k=2) == ["a", "b"]


def test_city_filter_prefers_the_longest_matching_town_name() -> None:
    # Real names overlap: matching the shortest first would send a query about
    # Castelnaudary to Castelnau.
    inner = FakeStrategy(["a", "b"])
    cities = {"a": "Castelnau", "b": "Castelnaudary"}

    assert CityFilter(inner, cities).search("marche a Castelnaudary", k=2) == ["b"]


def test_rerank_reorders_candidates_by_score() -> None:
    inner = FakeStrategy(["a", "b", "c"])

    def scorer(query: str, uids: list[str]) -> list[float]:
        return [0.1, 0.9, 0.5][: len(uids)]

    assert Rerank(inner, scorer).search("q", k=2) == ["b", "c"]


def test_rerank_returns_the_inner_order_when_the_scorer_is_flat() -> None:
    inner = FakeStrategy(["a", "b"])

    assert Rerank(inner, lambda q, uids: [1.0] * len(uids)).search("q", k=2) == ["a", "b"]
