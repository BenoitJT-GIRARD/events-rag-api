from events_rag.evaluation.report import render_table


def _result(name: str, recall: float | None = 0.8, error: str | None = None) -> dict:
    metrics = None if error else {"n": 20, "recall@1": 0.5, "recall@5": recall, "mrr@10": 0.6}
    return {
        "name": name,
        "variant": "baseline",
        "note": "a note",
        "metrics": metrics,
        "median_latency_ms": None if error else 12.5,
        "error": error,
    }


def test_table_has_one_row_per_configuration() -> None:
    table = render_table([_result("bm25-only"), _result("dense-baseline")])

    # Header, separator, then one row per configuration.
    assert len(table.splitlines()) == 4
    assert "bm25-only" in table
    assert "dense-baseline" in table


def test_failed_configuration_shows_its_error_rather_than_disappearing() -> None:
    table = render_table([_result("broken", error="RuntimeError: index missing")])

    assert "broken" in table
    assert "RuntimeError: index missing" in table


def test_best_recall_is_marked() -> None:
    table = render_table([_result("weak", recall=0.4), _result("strong", recall=0.9)])

    strong_row = next(line for line in table.splitlines() if "strong" in line)
    weak_row = next(line for line in table.splitlines() if "weak" in line)

    assert "**0.9**" in strong_row
    assert "**" not in weak_row.split("|")[4]
