"""The scripts a reader is told to run, run — in their own process, with nothing mocked.

Two of them need no API key, and those are the two exercised here: the one that redraws the
published figure from the published results, and the one that refuses to start without a key.
Everything else in `scripts/` embeds the corpus or calls the model, so it cannot run in a
suite that is offline by construction; `docs/evaluation-protocol.md` says what it costs and
how to run it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.system, pytest.mark.claim]

ROOT = Path(__file__).resolve().parents[2]


def run(script: str, cwd: Path, *arguments: str, **env: str) -> subprocess.CompletedProcess:
    import os

    environment = {**os.environ, "PYTHONPATH": str(cwd / "src"), **env}
    return subprocess.run(
        [sys.executable, script, *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_published_figure_is_what_the_script_draws(tmp_path: Path) -> None:
    """Redraw the figure in a copy of the repository and compare it to the committed one.

    This is the whole claim the figure makes: that it comes from the results file next to it.
    A figure that no longer matches its data is the failure this catches, and it catches it
    without a key, because the drawing step reads a file rather than a model.
    """
    workspace = tmp_path / "repo"
    for directory in ("src", "scripts", "reports", "data"):
        shutil.copytree(ROOT / directory, workspace / directory)
    shutil.copy(ROOT / "pyproject.toml", workspace / "pyproject.toml")

    done = run("scripts/plot_ablation.py", workspace)

    assert done.returncode == 0, done.stderr
    redrawn = (workspace / "reports" / "figures" / "ablation.svg").read_text(encoding="utf-8")
    published = (ROOT / "reports" / "figures" / "ablation.svg").read_text(encoding="utf-8")
    assert redrawn == published


def test_building_the_index_refuses_without_a_key(tmp_path: Path) -> None:
    """The documented failure, in the documented place: named variable, non-zero exit."""
    workspace = tmp_path / "repo"
    for directory in ("src", "scripts", "data"):
        shutil.copytree(ROOT / directory, workspace / directory)
    shutil.copy(ROOT / "pyproject.toml", workspace / "pyproject.toml")

    done = run("scripts/build_index.py", workspace, EVENTS_RAG_MISTRAL_API_KEY="")

    assert done.returncode != 0
    assert "EVENTS_RAG_MISTRAL_API_KEY" in done.stderr


def test_the_published_capture_matches_its_manifest() -> None:
    """A screenshot drifts silently: the page changes, the image stays, and only the
    fingerprint in the manifest says so. `--check` takes nothing and reports."""
    done = run("scripts/capture.py", ROOT, "--check")

    assert done.returncode == 0, done.stdout + done.stderr


def test_the_corpus_profile_is_what_the_script_counts(tmp_path: Path) -> None:
    """`data/raw/SOURCE.md` once claimed 415 towns for a corpus of 414. Now it cites a file."""
    import json

    workspace = tmp_path / "repo"
    for directory in ("src", "scripts", "data"):
        shutil.copytree(ROOT / directory, workspace / directory)
    shutil.copy(ROOT / "pyproject.toml", workspace / "pyproject.toml")

    done = run("scripts/profile_corpus.py", workspace)

    assert done.returncode == 0, done.stderr
    profile = workspace / "reports" / "corpus_profile.json"
    recounted = json.loads(profile.read_text(encoding="utf-8"))
    published = json.loads((ROOT / "reports" / "corpus_profile.json").read_text(encoding="utf-8"))
    assert recounted == published


def test_the_difficulty_of_the_question_set_is_what_the_script_measures(tmp_path: Path) -> None:
    import json

    workspace = tmp_path / "repo"
    for directory in ("src", "scripts", "data"):
        shutil.copytree(ROOT / directory, workspace / directory)
    shutil.copy(ROOT / "pyproject.toml", workspace / "pyproject.toml")

    done = run("scripts/measure_difficulty.py", workspace)

    assert done.returncode == 0, done.stderr
    measured = workspace / "reports" / "difficulty.json"
    remeasured = json.loads(measured.read_text(encoding="utf-8"))
    published = json.loads((ROOT / "reports" / "difficulty.json").read_text(encoding="utf-8"))
    assert remeasured == published
