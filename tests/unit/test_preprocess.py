from events_rag.ingestion.preprocess import event_to_document, strip_html


def test_strip_html_removes_tags() -> None:
    html = "<p>Bonjour <strong>Montpellier</strong></p>"
    assert strip_html(html) == "Bonjour Montpellier"


def test_event_to_document_maps_opendatasoft_fields() -> None:
    event = {
        "uid": "123",
        "title_fr": "Concert à Montpellier",
        "description_fr": "Un super concert",
        "longdescription_fr": "<p>Avec des artistes locaux</p>",
        "conditions_fr": "Gratuit",
        "location_city": "Montpellier",
        "location_name": "Opéra",
        "location_address": "Place de la Comédie",
        "firstdate_begin": "2026-04-01T18:00:00+00:00",
        "keywords_fr": ["musique", "concert"],
        "canonicalurl": "https://example.com/event",
    }

    doc = event_to_document(event)

    assert doc["uid"] == "123"
    assert "Concert à Montpellier" in doc["text"]
    assert doc["metadata"]["city"] == "Montpellier"
    assert doc["metadata"]["location_name"] == "Opéra"
    assert doc["metadata"]["date"] == "2026-04-01T18:00:00+00:00"
