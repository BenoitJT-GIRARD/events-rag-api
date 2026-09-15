"""Every artefact this repository publishes is written with Unix line endings.

Without `newline=""` a run on Windows writes CRLF and the same run on Linux writes LF. The
file then differs from the committed one on every line, and `scripts/smoke.py`, whose whole
job is to prove that the published files are the files the code produces, reports a
difference that is nothing but the platform it ran on.
"""

from __future__ import annotations

import subprocess

import pytest

from events_rag.utils.paths import ROOT_DIR

pytestmark = pytest.mark.claim

#: What a reader opens: the published results, the tables, the manifests. Not the corpus,
#: which is committed as it was fetched, and not the coverage report, which its own tool
#: writes.
PUBLISHED = ("reports/",)
EXCLUDED = ("reports/coverage/",)


def tracked_artefacts() -> list[str]:
    done = subprocess.run(
        ["git", "ls-files", "--", *PUBLISHED],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        rel
        for rel in done.stdout.split()
        if rel.endswith((".json", ".md", ".svg")) and not rel.startswith(EXCLUDED)
    ]


def test_there_are_artefacts_to_check() -> None:
    assert len(tracked_artefacts()) >= 5


def test_no_published_artefact_carries_a_carriage_return() -> None:
    """One test over every artefact, and not one test per artefact.

    A `parametrize` over a computed list cannot be counted without running pytest, and the
    README publishes how many tests this repository has. The failure message names every
    offending file, so the granularity that matters survives.
    """
    with_cr = [rel for rel in tracked_artefacts() if b"\r" in (ROOT_DIR / rel).read_bytes()]

    assert with_cr == []
