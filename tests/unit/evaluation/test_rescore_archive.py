"""The correction of the audit, replayed on answers that cannot be asked again.

The scorer of 2026-09-02 recorded all five out-of-corpus questions as wrong answers, because
it looked for the noun « événement » in a refusal that uses the noun of the question. The
answers are archived; the correction is checked by scoring them again.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from events_rag.evaluation.evaluate import score_answer
from events_rag.evaluation.ragas_eval import metric_summary
from events_rag.utils.paths import ROOT_DIR

SPEC = importlib.util.spec_from_file_location(
    "rescore_archive", ROOT_DIR / "scripts" / "rescore_archive.py"
)
rescore_archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rescore_archive)


@pytest.fixture(scope="module")
def archive() -> dict:
    return json.loads(
        (ROOT_DIR / "reports" / "evaluation_2026-09-02.json").read_text(encoding="utf-8")
    )


def test_the_five_refusals_the_old_scorer_missed_are_all_recognised(archive: dict) -> None:
    negatives = [case for case in archive["results"] if case["case_type"] == "negative"]

    assert len(negatives) == 5
    assert [case["label"] for case in negatives] == ["correct"] * 5


def test_the_published_summary_is_kept_beside_the_corrected_one(archive: dict) -> None:
    """The before is what makes the after readable, so the archive carries both."""

    assert archive["published_summary"]["correct"] == 22
    assert archive["published_summary"]["incorrect"] == 5
    assert archive["summary"]["correct"] == 27
    assert archive["summary"]["incorrect"] == 0


def test_coverage_is_averaged_over_the_twenty_positive_cases(archive: dict) -> None:
    assert archive["published_summary"]["avg_keyword_coverage"] == 0.908
    assert archive["summary"]["avg_keyword_coverage_positive"] == 0.938
    assert archive["summary"]["keyword_coverage_n"] == 20


def test_rescoring_the_archive_twice_changes_nothing(archive: dict) -> None:
    """The script rewrites the file it reads, so it has to be idempotent."""

    once = rescore_archive.rescore(archive)
    twice = rescore_archive.rescore(once)

    assert once == twice
    assert once["results"] == archive["results"]


@pytest.mark.parametrize(
    "answer",
    [
        "Aucune information concernant des opéras à l'Opéra Bastille.",
        "Aucun festival de cirque à Lyon n'est mentionné.",
        "Aucune exposition d'art contemporain ne figure dans le contexte.",
        "Il n'y a pas d'événement de ce type dans le corpus.",
        "Je ne dispose pas d'information sur ce sujet.",
    ],
)
def test_a_refusal_phrased_in_the_noun_of_the_question_is_not_a_wrong_answer(answer: str) -> None:
    """The defect itself: the old list of literal nouns matched none of these five."""

    coverage, label = score_answer(answer, "negative", [], has_city_match=False)

    assert label == "correct"
    assert coverage is None


def test_a_metric_is_read_from_the_column_ragas_writes_it_into() -> None:
    """The defect the recomputation surfaced: two spellings of one metric, and they differed.

    `ContextRelevancy` is served by the NVIDIA context-relevance metric, whose column is
    `nv_context_relevance`. The summary read `context_relevancy`, found nothing, and
    published null for a value measured on all thirty rows.
    """
    rows = [{"nv_context_relevance": 0.5}, {"nv_context_relevance": 1.0}]

    summary = metric_summary(rows)

    assert summary["avg_context_relevancy"] == 0.75
    assert summary["n_context_relevancy"] == 2


def test_a_mean_carries_the_number_of_rows_it_is_taken_over(archive: dict) -> None:
    """`context_precision` came out on 26 of the 30 answers, and the file says so."""

    ragas = json.loads((ROOT_DIR / "reports" / "ragas_2026-09-02.json").read_text(encoding="utf-8"))

    assert ragas["summary"]["n_context_precision"] == 26
    assert ragas["summary"]["n_answer_relevancy"] == 0
    assert ragas["summary"]["avg_answer_relevancy"] is None
    assert ragas["published_summary"]["avg_context_relevancy"] is None
    assert ragas["summary"]["avg_context_relevancy"] == 0.658


def test_resummarizing_the_ragas_archive_twice_changes_nothing() -> None:
    payload = json.loads(
        (ROOT_DIR / "reports" / "ragas_2026-09-02.json").read_text(encoding="utf-8")
    )

    once = rescore_archive.resummarize_ragas(payload)

    assert rescore_archive.resummarize_ragas(once) == once
