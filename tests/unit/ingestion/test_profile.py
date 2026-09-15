"""Counting the corpus rather than remembering it.

`data/raw/SOURCE.md` claimed 415 towns for a corpus that holds 414: a number typed once,
never re-counted, and wrong by the time anyone checked. Everything a document says about the
corpus is now counted here and written to a file the document cites.
"""

from __future__ import annotations

from events_rag.ingestion.profile import profile_corpus

CORPUS = [
    {"text": "Concert à Albi", "metadata": {"city": "Albi"}},
    {"text": "Exposition à Albi", "metadata": {"city": "Albi"}},
    {"text": "Marché à Nîmes", "metadata": {"city": "Nîmes"}},
    {"text": "Événement sans ville", "metadata": {}},
]


def test_the_towns_are_counted_distinctly() -> None:
    assert profile_corpus(CORPUS)["distinct_towns"] == 2


def test_a_record_without_a_town_is_counted_as_such_not_as_a_town() -> None:
    profile = profile_corpus(CORPUS)

    assert profile["events_without_a_town"] == 1
    assert profile["events"] == 4


def test_the_busiest_towns_come_out_in_order() -> None:
    busiest = profile_corpus(CORPUS, busiest=2)["busiest_towns"]

    assert busiest[0] == {"town": "Albi", "events": 2}
    assert len(busiest) == 2


def test_the_mean_length_is_over_the_texts_that_exist() -> None:
    profile = profile_corpus([{"text": "abcd"}, {"text": "ab"}])

    assert profile["mean_text_characters"] == 3


def test_an_empty_corpus_does_not_divide_by_zero() -> None:
    profile = profile_corpus([])

    assert profile["events"] == 0
    assert profile["mean_text_characters"] == 0
