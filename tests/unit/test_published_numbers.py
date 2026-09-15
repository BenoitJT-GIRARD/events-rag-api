"""Every number the README publishes is the number the results file holds.

A table retyped by hand drifts from its source without anyone noticing: before this test, the
latency column of the README came from an earlier run than the one published beside it, and
the two disagreed by a factor of four on the fastest row.

The withdrawn row is part of the contract too. `reports/errata.json` says which measurement
can no longer be read as one, and the README must show that row as withdrawn rather than
quietly keeping a number the code no longer produces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.claim

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "ablation_results.json"
ERRATA = ROOT / "reports" / "errata.json"
README = ROOT / "README.md"

ROW = re.compile(r"^\| `(?P<name>[^`]+)` \| (?P<rest>.+) \|$", re.M)


def published_rows() -> dict[str, list[str]]:
    rows = {}
    for match in ROW.finditer(README.read_text(encoding="utf-8")):
        cells = [cell.strip() for cell in match.group("rest").split("|")]
        rows[match.group("name")] = cells
    return rows


def withdrawn() -> set[str]:
    entries = json.loads(ERRATA.read_text(encoding="utf-8"))["entries"]
    return {e["row"] for e in entries if e["artefact"] == "reports/ablation_results.json"}


def measured() -> dict[str, dict]:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    return {result["name"]: result for result in payload["results"]}


def test_every_configuration_that_ran_has_a_row_in_the_readme() -> None:
    assert set(measured()) <= set(published_rows())


def test_each_row_carries_the_numbers_of_the_results_file() -> None:
    rows = published_rows()
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
    rows = published_rows()
    for name in withdrawn():
        assert all(cell.strip("*") == "withdrawn" for cell in rows[name][1:]), name


def test_the_sample_size_the_readme_states_is_the_one_that_was_measured() -> None:
    sizes = {r["metrics"]["n"] for r in measured().values() if r["metrics"]}
    assert len(sizes) == 1
    assert f"n = {sizes.pop()} questions" in README.read_text(encoding="utf-8")
