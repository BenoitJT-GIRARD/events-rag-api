"""The context handed to the model carries the metadata, and the sources come back serialisable.

The sources are what lets a reader check the answer against the events it came from.
"""

from langchain_core.documents import Document

from events_rag.rag.service import build_sources, format_context


def test_format_context_includes_metadata_and_content() -> None:
    documents = [
        Document(
            page_content="Contenu de test",
            metadata={
                "uid": "1",
                "title": "Concert jazz",
                "city": "Montpellier",
                "date": "2026-03-10",
                "location_name": "Opéra",
                "location_address": "Place de la Comédie",
                "conditions": "Gratuit",
            },
        )
    ]

    context = format_context(documents)

    assert "Concert jazz" in context
    assert "Montpellier" in context
    assert "Contenu de test" in context


def test_build_sources_returns_serializable_source_items() -> None:
    documents = [
        Document(
            page_content="Test",
            metadata={
                "uid": "1",
                "title": "Concert jazz",
                "city": "Montpellier",
                "date": "2026-03-10",
            },
        )
    ]

    sources = build_sources(documents)

    assert len(sources) == 1
    assert sources[0]["uid"] == "1"
    assert sources[0]["title"] == "Concert jazz"
    assert sources[0]["city"] == "Montpellier"
