"""Measure how much a question borrows from the document it is meant to retrieve.

A benchmark whose questions quote their source document measures string matching, not
retrieval. This puts a number on it, so the claim that a set is hard stops being an
assertion and becomes a figure that can be reported alongside the scores.
"""

import re
import unicodedata

# Four characters or more skips most French function words without needing a stopword
# list, which would be one more thing to justify and maintain.
MIN_TOKEN_LENGTH = 4


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def content_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", _fold(text)) if len(token) >= MIN_TOKEN_LENGTH
    }


def lexical_overlap(question: str, document: str) -> float:
    """Fraction of the question's content words that also appear in the document."""
    asked = content_tokens(question)
    if not asked:
        return 0.0
    return len(asked & content_tokens(document)) / len(asked)


def measure_question_set(cases: list[dict], events: dict[str, str]) -> dict:
    """The overlap of every question that names the event it was written from.

    A negative case has no source event by construction — it asks about something the corpus
    does not hold — so it contributes nothing to the mean rather than a free zero.
    """
    measured = [
        {
            "id": case["id"],
            "source_uid": case["source_uid"],
            "lexical_overlap": round(
                lexical_overlap(case["question"], events[case["source_uid"]]), 3
            ),
        }
        for case in cases
        if case.get("source_uid") and case.get("source_uid") in events
    ]
    overlaps = [row["lexical_overlap"] for row in measured]
    return {
        "cases_measured": len(measured),
        "mean_lexical_overlap": round(sum(overlaps) / len(overlaps), 2) if overlaps else None,
        "definition": (
            "Fraction of a question's content words (four characters or more, accents "
            "folded) that appear verbatim in the event it was written from."
        ),
        "per_case": measured,
    }
