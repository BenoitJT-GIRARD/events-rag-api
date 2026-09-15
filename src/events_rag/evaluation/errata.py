"""Which published rows can no longer be read as a measurement.

A result that turns out to have measured the wrong thing is withdrawn, and the withdrawal is
written down in ``reports/errata.json``: the artefact, the row, the reason, the fix, and the
command that would measure it again. The figure and the README both need to know that list,
and a list computed twice is a list that ends up differing, so it is computed here.

An entry that carries ``resolved_on`` has been measured again. It stays in the errata, since
the correction is part of what the number means, and its row stops being withdrawn.
"""

from __future__ import annotations

import json
from pathlib import Path

FILENAME = "errata.json"


def entries(reports_dir: Path) -> list[dict]:
    errata = Path(reports_dir) / FILENAME
    if not errata.is_file():
        return []
    return json.loads(errata.read_text(encoding="utf-8")).get("entries", [])


def withdrawn_rows(reports_dir: Path, artefact: str) -> set[str]:
    """The rows of ``artefact`` that are withdrawn and not yet measured again."""
    return {
        entry["row"]
        for entry in entries(reports_dir)
        if entry.get("artefact") == artefact and entry.get("row") and not entry.get("resolved_on")
    }
