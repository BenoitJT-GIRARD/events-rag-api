"""recall@k, reciprocal rank, and the deduplication that has to happen before the cut at k.

Several chunks of one event can be retrieved; counting them separately would inflate recall
at every k. The absent-target cases are here too, because a metric that only ever sees a
hit is a metric nobody has tested.
"""

import pytest

from events_rag.evaluation import retrieval
from events_rag.evaluation.retrieval import (
    aggregate,
    dedupe_uids,
    recall_at_k,
    reciprocal_rank,
)


def test_dedupe_keeps_first_occurrence_order() -> None:
    assert dedupe_uids(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_is_one_when_target_is_within_k() -> None:
    assert recall_at_k(["x", "y", "target"], "target", k=3) == 1.0


def test_recall_is_zero_when_target_falls_outside_k() -> None:
    assert recall_at_k(["x", "y", "target"], "target", k=2) == 0.0


def test_recall_deduplicates_before_cutting_at_k() -> None:
    # The same event yields several chunks. Without dedup "target" sits at rank 3
    # and recall@2 would be 0.0; after dedup it sits at rank 2.
    assert recall_at_k(["x", "x", "target"], "target", k=2) == 1.0


def test_reciprocal_rank_is_the_inverse_of_the_first_hit() -> None:
    assert reciprocal_rank(["x", "target", "y"], "target") == 0.5


def test_reciprocal_rank_is_zero_when_target_is_absent() -> None:
    assert reciprocal_rank(["x", "y"], "target") == 0.0


def test_aggregate_reports_each_k_and_the_sample_size() -> None:
    rows = [
        {"target": "a", "retrieved": ["a", "b"]},
        {"target": "b", "retrieved": ["c", "b"]},
    ]

    summary = aggregate(rows, ks=(1, 2))

    assert summary["n"] == 2
    assert summary["recall@1"] == 0.5
    assert summary["recall@2"] == 1.0
    assert summary["mrr@10"] == 0.75


# --- The comparison is paired: seven configurations, the same twenty questions ---


def test_the_rank_of_the_target_is_one_based_and_deduplicated() -> None:
    """Two chunks of the same event occupy two positions; the event holds one rank."""

    assert retrieval.target_rank(["a", "a", "b"], "b") == 2
    assert retrieval.target_rank(["a", "b"], "a") == 1
    assert retrieval.target_rank(["a", "b"], "z") is None


def test_only_the_discordant_questions_carry_information() -> None:
    """Both right, or both wrong, says nothing about which configuration is better."""

    both = [True, True, False, False]

    result = retrieval.mcnemar_exact(both, both)

    assert result["discordant"] == 0
    assert result["p_value"] == 1.0


def test_one_question_of_difference_decides_nothing() -> None:
    """The gap the README refuses to claim: 0.95 against 0.90 at n = 20."""

    city_filter = [True] * 19 + [False]
    dense = [True] * 18 + [False, False]

    result = retrieval.mcnemar_exact(city_filter, dense)

    assert result["only_first"] == 1
    assert result["only_second"] == 0
    assert result["p_value"] == 1.0


def test_a_clean_sweep_of_discordant_pairs_is_decided() -> None:
    """Six questions won and none lost: the test says what twenty questions can say."""

    winner = [True] * 6 + [False] * 4
    loser = [False] * 6 + [False] * 4

    result = retrieval.mcnemar_exact(winner, loser)

    assert result["discordant"] == 6
    assert result["p_value"] < 0.05


def test_two_runs_of_different_length_are_not_a_pair() -> None:
    with pytest.raises(ValueError, match="same questions"):
        retrieval.mcnemar_exact([True, False], [True])
