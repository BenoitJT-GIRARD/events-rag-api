"""A metric that did not compute is reported as missing, and survives serialisation as null.

Two of the five RAGAS metrics return NaN on this setup. The failure to avoid is the one where
a NaN reaches the results file: it either serialises as the bare token `NaN`, which no JSON
reader accepts, or it is coerced to zero, and a zero in a results table reads as a measured
score of nothing rather than as a gap.
"""

from __future__ import annotations

import json
import math

from events_rag.evaluation import ragas_eval


def test_a_mean_ignores_the_metrics_that_did_not_compute() -> None:
    assert ragas_eval.safe_mean([0.8, float("nan"), None, 0.6]) == 0.7


def test_a_mean_of_nothing_measurable_is_none_not_zero() -> None:
    """Zero is a score. Nothing measured is not a score, and the table must say so."""
    assert ragas_eval.safe_mean([float("nan"), None]) is None


def test_a_missing_metric_serialises_as_null() -> None:
    payload = {"faithfulness": float("nan"), "precision": 0.42, "nested": [float("inf"), 1.0]}

    written = json.dumps(payload, cls=ragas_eval.NaNSafeEncoder)

    assert json.loads(written) == {
        "faithfulness": None,
        "precision": 0.42,
        "nested": [None, 1.0],
    }
    assert "NaN" not in written


def test_the_case_type_summary_counts_what_the_set_actually_holds() -> None:
    cases = [
        {"case_type": "positive"},
        {"case_type": "negative"},
        {"case_type": "positive"},
        {},
    ]

    assert ragas_eval.build_case_type_summary(cases) == {"positive": 3, "negative": 1}


def test_a_dataset_that_is_not_a_list_is_refused(tmp_path) -> None:
    path = tmp_path / "reference_qa.json"
    path.write_text(json.dumps({"cases": []}), encoding="utf-8")

    try:
        ragas_eval.load_reference_dataset(path)
    except ValueError as error:
        assert "JSON list" in str(error)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("a mapping was accepted as an evaluation set")


def test_an_infinite_value_is_treated_like_a_missing_one() -> None:
    assert ragas_eval.safe_mean([math.inf, 0.5]) == 0.5
