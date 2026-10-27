"""`/rebuild` refuses without its token, and a rebuild it runs is what the next question reads.

The route re-embeds the whole corpus through a paid API. An open one is a bill anyone can
run up.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from events_rag.api import main
from events_rag.api.main import app
from events_rag.config import get_settings
from events_rag.rag import retriever

pytestmark = pytest.mark.integration


def test_rebuild_requires_token() -> None:
    with TestClient(app) as client:
        response = client.post("/rebuild", json={"token": "wrong-token"})

    assert response.status_code in {403, 500}


class _LoadedStore:
    """What `FAISS.load_local` hands back, one new object per read from disk."""

    reads = 0

    @classmethod
    def load_local(cls, **_kwargs) -> object:
        cls.reads += 1
        return object()


@pytest.fixture
def _cached_store(tmp_path: Path, monkeypatch):
    """A store loaded once through the real cache of `load_vectorstore`.

    Two doubles, both standing in for the paid embedding API: reading a FAISS index needs an
    embeddings object built with the Mistral key, and the rebuild itself embeds the corpus.
    """
    settings = get_settings()
    (tmp_path / settings.faiss_index_name).mkdir()
    monkeypatch.setattr(settings, "index_dir", tmp_path)
    monkeypatch.setattr(settings, "rebuild_token", "a-token-for-this-test")
    monkeypatch.setattr(retriever, "FAISS", _LoadedStore)
    monkeypatch.setattr(retriever, "build_embeddings", lambda: None)
    monkeypatch.setattr(
        main,
        "build_and_save_index",
        lambda: {
            "status": "success",
            "indexed_documents": 1,
            "indexed_chunks": 1,
            "index_path": str(tmp_path / settings.faiss_index_name),
        },
    )
    retriever.load_vectorstore.cache_clear()
    yield retriever.load_vectorstore()
    retriever.load_vectorstore.cache_clear()


def test_the_question_after_a_rebuild_reads_the_new_index(_cached_store) -> None:
    with TestClient(app) as client:
        response = client.post("/rebuild", json={"token": "a-token-for-this-test"})

    assert response.status_code == 200
    assert retriever.load_vectorstore() is not _cached_store
