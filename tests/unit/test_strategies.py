from langchain_core.documents import Document

from events_rag.evaluation.strategies import BM25Search, DenseSearch


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
