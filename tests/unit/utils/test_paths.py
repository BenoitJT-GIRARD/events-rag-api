"""Where the files are is decided once, and the answer survives being installed.

Counting directories up from a module works in an editable checkout and is wrong the moment
the package is installed as a wheel: it would then write the index, the reports and the
figures somewhere inside site-packages, without a word.
"""

from __future__ import annotations

import os
from pathlib import Path

from events_rag.utils import paths


def test_the_root_is_the_directory_that_holds_the_pyproject() -> None:
    assert (paths.ROOT_DIR / "pyproject.toml").is_file()


def test_every_directory_hangs_off_the_root() -> None:
    for directory in (
        paths.DATA_DIR,
        paths.RAW_DIR,
        paths.QUESTIONS_DIR,
        paths.REPORTS_DIR,
        paths.FIGURES_DIR,
        paths.INDEX_DIR,
        paths.VAR_DIR,
    ):
        assert paths.ROOT_DIR in directory.parents


def test_the_index_is_not_inside_the_data_the_program_reads() -> None:
    """`data/` is what the pipeline consumes; the index is what a run rebuilds from it."""
    assert paths.DATA_DIR not in paths.INDEX_DIR.parents
    assert paths.VAR_DIR in paths.INDEX_DIR.parents


def test_the_override_is_read_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))

    assert paths._find_root() == tmp_path.resolve()


def test_the_variable_is_named_after_the_package() -> None:
    assert paths.ROOT_ENV == "EVENTS_RAG_ROOT"
    assert os.environ.get(paths.ROOT_ENV) in (None, "")


def test_the_corpus_is_logged_by_its_place_in_the_project() -> None:
    """`indexer.raw_documents_loaded` says which file was read, not which disk held it."""
    assert paths.rel(paths.REPORTS_DIR / "figures" / "roc.png") == "reports/figures/roc.png"


def test_a_corpus_opened_from_somewhere_else_keeps_its_whole_path(tmp_path) -> None:
    """An index built from a corpus outside the tree still has to say which file it read."""
    elsewhere = tmp_path / "events.json"

    assert paths.rel(elsewhere) == elsewhere.resolve().as_posix()
