"""The chunking variants build the indexes the ablation compares, and are told apart.

Each variant carries a fingerprint. Two variants that produce the same fingerprint would be
compared as different configurations while sharing one index, and the ablation would be
measuring nothing — so the fingerprint has to move when the chunking does.
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
    # Title, venue, city and date live in metadata and never reach the embedding, so a
    # query naming a venue has nothing to match against. This variant puts them in text.
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
