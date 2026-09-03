"""Raw records become documents, and a long document is split rather than truncated."""

from langchain_core.documents import Document

from events_rag.rag.indexer import split_documents, to_langchain_documents


def test_to_langchain_documents_creates_documents() -> None:
    raw_items = [
        {
            "uid": "123",
            "text": "Titre: Concert\n\nDescription: Musique live",
            "metadata": {
                "uid": "123",
                "title": "Concert",
                "city": "Montpellier",
            },
        }
    ]

    docs = to_langchain_documents(raw_items)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["uid"] == "123"
    assert "Concert" in docs[0].page_content


def test_split_documents_splits_long_document() -> None:
    docs = [
        Document(
            page_content="A" * 1200,
            metadata={"uid": "abc", "title": "Long event"},
        )
    ]

    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.metadata["uid"] == "abc" for chunk in chunks)
