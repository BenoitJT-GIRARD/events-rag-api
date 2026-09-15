"""What pins a chunking variant: its fingerprint, and what the header variant puts in text.

Two variants sharing a fingerprint would share an index and be reported as two
configurations, so every field that changes the chunking has to change the fingerprint. The
header case is checked on the text itself, since that is where the difference has to land.
"""

from langchain_core.documents import Document

from events_rag.evaluation.indexes import (
    ChunkingVariant,
    split_for_variant,
    variant_fingerprint,
)


def _events() -> list[Document]:
    long_text = "phrase. " * 300
    return [Document(page_content=long_text, metadata={"uid": "a"})]


def test_fingerprint_is_stable_and_distinguishes_variants() -> None:
    a = ChunkingVariant("baseline", 800, 120)
    b = ChunkingVariant("one-chunk", None, None)

    assert variant_fingerprint(a) == variant_fingerprint(ChunkingVariant("baseline", 800, 120))
    assert variant_fingerprint(a) != variant_fingerprint(b)


def test_one_chunk_per_event_leaves_documents_untouched() -> None:
    documents = _events()

    chunks = split_for_variant(documents, ChunkingVariant("one-chunk", None, None))

    assert len(chunks) == 1
    assert chunks[0].page_content == documents[0].page_content


def test_sized_variant_splits_a_long_event_into_several_chunks() -> None:
    chunks = split_for_variant(_events(), ChunkingVariant("baseline", 800, 120))

    assert len(chunks) > 1
    assert all(chunk.metadata["uid"] == "a" for chunk in chunks)


def test_header_variant_embeds_the_metadata_in_the_indexed_text() -> None:
    # Title, venue, city and date stay out of the vector, so the header variant is the only
    # one where a query naming a venue has anything to match.
    documents = [
        Document(
            page_content="soiree musicale",
            metadata={
                "uid": "a",
                "title": "Le Vieux Biclou",
                "location_name": "Opera Comedie",
                "city": "Montpellier",
                "date": "2026-05-25",
            },
        )
    ]

    chunks = split_for_variant(documents, ChunkingVariant("header", None, None, header=True))

    assert "Le Vieux Biclou" in chunks[0].page_content
    assert "Opera Comedie" in chunks[0].page_content
    assert "soiree musicale" in chunks[0].page_content
    assert chunks[0].metadata["uid"] == "a"


def test_header_changes_the_fingerprint() -> None:
    plain = ChunkingVariant("plain", None, None)
    with_header = ChunkingVariant("header", None, None, header=True)

    assert variant_fingerprint(plain) != variant_fingerprint(with_header)
