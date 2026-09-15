"""The question set is sampled under a fixed seed, and that is what makes it comparable.

Taking the head of the file biased the set toward one slice of the corpus. Sampling fixes the
bias; fixing the seed fixes something else — two ablation configurations compared on two
different question sets are not compared at all.
"""

from __future__ import annotations

import random

from events_rag.evaluation.difficulty import measure_question_set
from events_rag.evaluation.generate_dataset import (
    MAX_EVENTS_PER_CATEGORY,
    select_events_for_category,
)

CORPUS = [{"text": f"Concert numéro {index} à Albi"} for index in range(40)] + [
    {"text": "Exposition de photographies"},
    {"text": "Marché de producteurs"},
]


def test_only_the_events_matching_a_keyword_are_eligible() -> None:
    chosen = select_events_for_category(CORPUS, ["exposition"], random.Random(0))

    assert [event["text"] for event in chosen] == ["Exposition de photographies"]


def test_the_same_seed_gives_the_same_questions() -> None:
    first = select_events_for_category(CORPUS, ["concert"], random.Random(0))
    second = select_events_for_category(CORPUS, ["concert"], random.Random(0))

    assert first == second


def test_two_seeds_give_two_sets_which_is_why_the_seed_is_fixed() -> None:
    first = select_events_for_category(CORPUS, ["concert"], random.Random(0))
    other = select_events_for_category(CORPUS, ["concert"], random.Random(7))

    assert first != other


def test_the_sample_is_capped_and_never_takes_the_head_of_the_file() -> None:
    chosen = select_events_for_category(CORPUS, ["concert"], random.Random(0))

    assert len(chosen) == MAX_EVENTS_PER_CATEGORY
    head = [{"text": f"Concert numéro {index} à Albi"} for index in range(MAX_EVENTS_PER_CATEGORY)]
    assert chosen != head


def test_a_category_with_fewer_matches_than_the_cap_keeps_them_all() -> None:
    chosen = select_events_for_category(CORPUS, ["marché"], random.Random(0))

    assert len(chosen) == 1


def test_only_the_questions_with_a_source_event_are_measured() -> None:
    """A negative case asks about something the corpus does not hold: nothing to overlap."""
    cases = [
        {"id": "q1", "question": "Un concert à Albi", "source_uid": "evt-1"},
        {"id": "q2", "question": "Un festival inexistant", "case_type": "negative"},
    ]
    events = {"evt-1": "Concert à Albi, salle municipale"}

    measured = measure_question_set(cases, events)

    assert measured["cases_measured"] == 1
    assert [row["id"] for row in measured["per_case"]] == ["q1"]


def test_the_mean_overlap_is_the_number_the_documents_cite() -> None:
    cases = [
        {"id": "q1", "question": "concert albi", "source_uid": "evt-1"},
        {"id": "q2", "question": "exposition nimes", "source_uid": "evt-2"},
    ]
    events = {"evt-1": "concert albi salle", "evt-2": "marche toulouse"}

    measured = measure_question_set(cases, events)

    assert measured["per_case"][0]["lexical_overlap"] == 1.0
    assert measured["per_case"][1]["lexical_overlap"] == 0.0
    assert measured["mean_lexical_overlap"] == 0.5


def test_a_set_with_no_measurable_case_has_no_mean() -> None:
    assert measure_question_set([], {})["mean_lexical_overlap"] is None
