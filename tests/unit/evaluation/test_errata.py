"""A withdrawn row stops being withdrawn the day it is measured again.

The errata keeps the history either way: what was wrong is part of what the number means.
"""

from __future__ import annotations

import json
from pathlib import Path

from events_rag.evaluation.errata import entries, withdrawn_rows

ARTEFACT = "reports/ablation_results.json"


def write(directory: Path, payload: dict) -> Path:
    (directory / "errata.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def test_an_unresolved_entry_withdraws_its_row(tmp_path: Path) -> None:
    reports = write(tmp_path, {"entries": [{"artefact": ARTEFACT, "row": "hybrid-rrf+rerank"}]})

    assert withdrawn_rows(reports, ARTEFACT) == {"hybrid-rrf+rerank"}


def test_a_resolved_entry_gives_its_row_back(tmp_path: Path) -> None:
    reports = write(
        tmp_path,
        {
            "entries": [
                {"artefact": ARTEFACT, "row": "hybrid-rrf+rerank", "resolved_on": "2026-09-15"}
            ]
        },
    )

    assert withdrawn_rows(reports, ARTEFACT) == set()
    assert len(entries(reports)) == 1


def test_an_entry_about_another_artefact_is_not_ours(tmp_path: Path) -> None:
    reports = write(tmp_path, {"entries": [{"artefact": "reports/difficulty.json", "row": "x"}]})

    assert withdrawn_rows(reports, ARTEFACT) == set()


def test_a_repository_without_errata_withdraws_nothing(tmp_path: Path) -> None:
    assert withdrawn_rows(tmp_path, ARTEFACT) == set()
    assert entries(tmp_path) == []
