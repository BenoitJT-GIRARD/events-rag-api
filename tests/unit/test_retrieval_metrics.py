"""recall@k, reciprocal rank, and the deduplication that has to happen before the cut at k.

Several chunks of one event can be retrieved; counting them separately would inflate recall
at every k. The absent-target cases are here too, because a metric that only ever sees a
hit is a metric nobody has tested.
"""

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
