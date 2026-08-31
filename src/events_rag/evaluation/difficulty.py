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
