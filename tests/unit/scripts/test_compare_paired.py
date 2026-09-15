"""The paired comparison the README publishes, computed from per-question ranks.

The counts in the table are what a reader checks first: how many questions one configuration
wins, how many it loses. They come from this function.
"""

from __future__ import annotations

import importlib.util

from events_rag.utils.paths import ROOT_DIR

SPEC = importlib.util.spec_from_file_location(
    "compare_paired", ROOT_DIR / "scripts" / "compare_paired.py"
)
compare_paired = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_paired)


def ranks(*values: int | None) -> list[dict]:
    return [{"uid": str(index), "rank": rank} for index, rank in enumerate(values)]


def test_a_target_below_k_is_not_a_hit() -> None:
    assert compare_paired.hits(ranks(1, 3, None, 6), 1) == [True, False, False, False]
    assert compare_paired.hits(ranks(1, 3, None, 6), 5) == [True, True, False, False]


def test_each_configuration_is_compared_at_both_depths() -> None:
    results = [
        {"name": "dense-baseline", "per_question": ranks(1, 1, 2, None)},
        {"name": "bm25-only", "per_question": ranks(4, None, 2, None)},
    ]

    comparison = compare_paired.compare(results)

    assert comparison["n_questions"] == 4
    assert [row["metric"] for row in comparison["comparisons"]] == ["recall@1", "recall@5"]
    at_one = comparison["comparisons"][0]
    assert at_one["configuration"] == "bm25-only"
    assert (at_one["won"], at_one["lost"]) == (0, 2)


def test_a_run_without_per_question_ranks_stops_the_script() -> None:
    import pytest

    with pytest.raises(SystemExit, match="rerun"):
        compare_paired.compare([{"name": "bm25-only", "per_question": []}])


def test_a_configuration_that_wins_and_loses_the_same_number_is_undecided() -> None:
    results = [
        {"name": "dense-baseline", "per_question": ranks(1, 3, 1, 3)},
        {"name": "hybrid-rrf", "per_question": ranks(3, 1, 1, 3)},
    ]

    at_one = compare_paired.compare(results)["comparisons"][0]

    assert (at_one["won"], at_one["lost"]) == (1, 1)
    assert at_one["p_value"] == 1.0
    assert at_one["decided_at_5_percent"] is False
