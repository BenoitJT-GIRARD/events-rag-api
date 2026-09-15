"""Every number the README publishes is the number the results file holds.

A table retyped by hand drifts from its source without anyone noticing: before this test, the
latency column of the README came from an earlier run than the one published beside it, and
the two disagreed by a factor of four on the fastest row.

The withdrawn row is part of the contract too. `reports/errata.json` says which measurement
can no longer be read as one, and the README must show that row as withdrawn for as long as
the entry is unresolved.
"""

from __future__ import annotations

import json
import re

import pytest

from events_rag.evaluation.errata import withdrawn_rows
from events_rag.utils.paths import ROOT_DIR as ROOT

pytestmark = pytest.mark.claim

RESULTS = ROOT / "reports" / "ablation_results.json"
PAIRED = ROOT / "reports" / "paired_comparison.json"
REPORTS = ROOT / "reports"
README = ROOT / "README.md"

ROW = re.compile(r"^\| `(?P<name>[^`]+)` \| (?P<rest>.+) \|$", re.M)


def table_after(header_fragment: str) -> dict[str, list[str]]:
    """The rows of the README table whose header carries `header_fragment`.

    The README publishes two tables keyed by configuration name, so a search over the whole
    document reads the second one over the first. Each test says which table it means.
    """
    text = README.read_text(encoding="utf-8")
    start = text.index(header_fragment)
    end = text.find("\n\n", start)
    rows: dict[str, list[str]] = {}
    for match in ROW.finditer(text[start:end]):
        rows[match.group("name")] = [cell.strip() for cell in match.group("rest").split("|")]
    return rows


def withdrawn() -> set[str]:
    return withdrawn_rows(REPORTS, "reports/ablation_results.json")


def measured() -> dict[str, dict]:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    return {result["name"]: result for result in payload["results"]}


def test_every_configuration_that_ran_has_a_row_in_the_readme() -> None:
    assert set(measured()) <= set(table_after("| Configuration | Chunking |"))


def test_each_row_carries_the_numbers_of_the_results_file() -> None:
    rows = table_after("| Configuration | Chunking |")
    for name, result in measured().items():
        if name in withdrawn():
            continue
        cells = rows[name]
        metrics = result["metrics"]
        assert cells[1] == f"{metrics['recall@1']:.2f}".replace("**", ""), name
        assert cells[2].strip("*") == f"{metrics['recall@5']:.2f}", name
        assert cells[3] == f"{metrics['mrr@10']:.3f}", name
        assert cells[4].strip("*") == f"{result['median_latency_ms']} ms", name


def test_a_withdrawn_row_publishes_no_number() -> None:
    rows = table_after("| Configuration | Chunking |")
    for name in withdrawn():
        assert all(cell.strip("*") == "withdrawn" for cell in rows[name][1:]), name


def test_the_sample_size_the_readme_states_is_the_one_that_was_measured() -> None:
    sizes = {r["metrics"]["n"] for r in measured().values() if r["metrics"]}
    assert len(sizes) == 1
    assert f"n = {sizes.pop()} questions" in README.read_text(encoding="utf-8")


def test_the_paired_table_carries_the_counts_of_the_comparison_file() -> None:
    """The second published table: won, lost and p, against the shipped configuration."""

    comparison = json.loads(PAIRED.read_text(encoding="utf-8"))
    at_one = {c["configuration"]: c for c in comparison["comparisons"] if c["metric"] == "recall@1"}
    rows = table_after("| Against `dense-baseline`, recall@1 |")

    assert set(rows) == set(at_one)
    for name, cells in rows.items():
        entry = at_one[name]
        assert cells[0] == str(entry["won"]), name
        assert cells[1] == str(entry["lost"]), name
        assert float(cells[2]) == pytest.approx(entry["p_value"], abs=5e-4), name


def test_every_configuration_is_compared_with_the_one_that_ships() -> None:
    comparison = json.loads(PAIRED.read_text(encoding="utf-8"))

    assert comparison["reference"] == "dense-baseline"
    assert comparison["n_questions"] == 20
    compared = {c["configuration"] for c in comparison["comparisons"]}
    assert compared == set(measured()) - {"dense-baseline"}
