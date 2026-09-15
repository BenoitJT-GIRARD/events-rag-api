"""Loading an index and retrieving from it, with the paid model replaced by a stub.

Three behaviours are decided here and none of them needs the network: the missing key is
refused with the name of the variable, a missing index says which script builds it, and the
number of documents asked for comes from the settings unless the caller overrides it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from events_rag.rag import retriever


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, question: str, k: int):
        self.calls.append((question, k))
        return [f"doc-{index}" for index in range(k)]


@pytest.fixture(autouse=True)
def _forget_cached_stores():
    retriever.load_vectorstore.cache_clear()
    retriever.build_embeddings.cache_clear()
    yield
    retriever.load_vectorstore.cache_clear()
    retriever.build_embeddings.cache_clear()


def test_a_missing_key_is_refused_with_the_name_of_the_variable(monkeypatch) -> None:
    settings = retriever.get_settings()
    monkeypatch.setattr(settings, "mistral_api_key", "")

    with pytest.raises(ValueError, match="EVENTS_RAG_MISTRAL_API_KEY"):
        retriever.build_embeddings()


def test_a_missing_index_names_the_script_that_builds_it(tmp_path: Path, monkeypatch) -> None:
    settings = retriever.get_settings()
    monkeypatch.setattr(settings, "index_dir", tmp_path / "absent")

    with pytest.raises(FileNotFoundError, match="scripts/build_index.py"):
        retriever.load_vectorstore()


def test_retrieval_asks_for_the_configured_number_of_documents(monkeypatch) -> None:
    store = FakeStore()
    monkeypatch.setattr(retriever, "load_vectorstore", lambda *a, **k: store)
    settings = retriever.get_settings()
    monkeypatch.setattr(settings, "retrieval_k", 4)

    documents = retriever.retrieve_documents("un concert à Albi")

    assert len(documents) == 4
    assert store.calls == [("un concert à Albi", 4)]


def test_an_explicit_top_k_wins_over_the_setting(monkeypatch) -> None:
    store = FakeStore()
    monkeypatch.setattr(retriever, "load_vectorstore", lambda *a, **k: store)

    retriever.retrieve_documents("une exposition", top_k=2)

    assert store.calls == [("une exposition", 2)]
