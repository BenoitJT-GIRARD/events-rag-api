"""What the index writes next to itself, and what it refuses to read.

The manifest is the only way a reader knows which corpus and which chunking produced the
vectors sitting in `var/faiss/`. It is written by the same call that saves the index, so the
two cannot describe different runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from events_rag.rag import indexer


class FakeStore:
    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save_local(self, folder_path: str) -> None:
        self.saved_to = folder_path
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        (Path(folder_path) / "index.faiss").write_bytes(b"vectors")


def test_the_manifest_names_the_corpus_and_the_chunking(tmp_path: Path, monkeypatch) -> None:
    settings = indexer.get_settings()
    monkeypatch.setattr(settings, "chunk_size", 800)
    monkeypatch.setattr(settings, "chunk_overlap", 120)
    store = FakeStore()

    output = indexer.save_faiss_index(
        vectorstore=store,
        index_dir=tmp_path,
        index_name="events_index",
        source_document_count=1000,
        chunk_count=2046,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_document_count"] == 1000
    assert manifest["chunk_count"] == 2046
    assert manifest["chunk_size"] == 800
    assert manifest["chunk_overlap"] == 120
    assert manifest["embedding_model"] == settings.embedding_model


def test_a_corpus_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"events": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON list"):
        indexer.load_raw_documents(path)


def test_an_event_with_no_text_is_dropped_rather_than_indexed_empty() -> None:
    documents = indexer.to_langchain_documents(
        [
            {"text": "Concert à Albi", "metadata": {"uid": "evt-1"}},
            {"text": "", "metadata": {"uid": "evt-2"}},
        ]
    )

    assert [doc.metadata["uid"] for doc in documents] == ["evt-1"]


def test_splitting_keeps_the_metadata_of_the_event_it_came_from() -> None:
    long_text = " ".join(f"phrase {index}." for index in range(400))
    documents = indexer.to_langchain_documents([{"text": long_text, "metadata": {"uid": "evt-1"}}])

    chunks = indexer.split_documents(documents, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert {chunk.metadata["uid"] for chunk in chunks} == {"evt-1"}
