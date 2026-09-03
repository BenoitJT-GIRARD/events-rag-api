"""Turning one API record into one document: text to embed, metadata to filter on.

The HTML strip matters more than it looks. Descriptions arrive with markup, and an
embedding of a `<p>` tag is an embedding of nothing.
"""

from html import unescape
from re import sub

from events_rag.logger import get_logger

logger = get_logger(__name__)


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    clean = sub(r"<[^>]+>", " ", text)
    clean = sub(r"\s+", " ", clean).strip()
    return unescape(clean)


def event_to_document(event: dict) -> dict:
    title = event.get("title_fr") or ""
    description = event.get("description_fr") or ""
    long_description = strip_html(event.get("longdescription_fr"))
    conditions = event.get("conditions_fr") or ""
    city = event.get("location_city") or ""
    location_name = event.get("location_name") or ""
    location_address = event.get("location_address") or ""
    first_date = event.get("firstdate_begin") or ""
    keywords = event.get("keywords_fr") or []

    keywords_text = ", ".join(keywords) if isinstance(keywords, list) else ""

    text = f"""
Titre: {title}

Description courte: {description}

Description longue: {long_description}

Conditions: {conditions}

Ville: {city}
Lieu: {location_name}
Adresse: {location_address}

Date de début: {first_date}

Mots-clés: {keywords_text}
""".strip()

    return {
        "uid": str(event.get("uid", "")),
        "text": text,
        "metadata": {
            "uid": str(event.get("uid", "")),
            "title": title,
            "city": city,
            "location_name": location_name,
            "location_address": location_address,
            "date": first_date,
            "conditions": conditions,
            "keywords": keywords,
            "canonicalurl": event.get("canonicalurl"),
        },
    }
