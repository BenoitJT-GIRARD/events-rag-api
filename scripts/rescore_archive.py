"""Recompute the summaries of the archived run of 2026-09-02 from its own answers.

Two files hold that run, and neither can be produced again: the questions it was asked from
were replaced by the hand-written set the same day, and the account that paid for the answers
has no chat quota left. What is still checkable is the arithmetic on top of them, and that is
what this script checks.

The heuristic archive is the interesting half. The methodological audit found that the
refusal detector looked for the literal noun « événement » while the model answers in the
noun of the question, so all five out-of-corpus questions were recorded as wrong answers when
the system had refused each one correctly. Scoring the archived answers again with the code
that replaced the detector is how that correction is verified.

    uv run python scripts/rescore_archive.py

Rewrites the two summaries, and the per-case labels of the heuristic archive. Both files
keep their `published_summary` untouched: the correction is only legible against what it
corrects.
"""

from __future__ import annotations

import json

from events_rag.evaluation.evaluate import score_answer, summarize
from events_rag.evaluation.ragas_eval import RAGAS_COLUMN, metric_summary
from events_rag.utils.paths import REPORTS_DIR

HEURISTIC = REPORTS_DIR / "evaluation_2026-09-02.json"
RAGAS = REPORTS_DIR / "ragas_2026-09-02.json"


def rescore(payload: dict) -> dict:
    """The heuristic archive, with every case scored by the current code.

    `city_match` is read back and not recomputed: it depends on the sources that run
    retrieved, which the archive carries, and no audit defect touched it.
    """
    results = []
    for case in payload["results"]:
        coverage, label = score_answer(
            case["answer"],
            case.get("case_type", "positive"),
            case.get("expected_keywords", []),
            bool(case.get("city_match")),
        )
        results.append(
            {
                **case,
                "keyword_coverage": round(coverage, 3) if coverage is not None else None,
                "label": label,
            }
        )
    return {**payload, "summary": summarize(results), "results": results}


def resummarize_ragas(payload: dict) -> dict:
    """The RAGAS archive, with its averages recomputed from the rows below them.

    Nothing here regrades: a grade is a model call. This recomputes the five means over the
    grades already in the file, which is what makes the published summary falsifiable.
    """
    rows = payload["results"]
    summary = {
        "count": len(rows),
        **metric_summary(rows),
        "by_case_type": payload["summary"].get("by_case_type"),
    }
    return {**payload, "summary": summary}


def _dump(payload: dict) -> str:
    """The exact bytes of an archive, so a rerun that changes nothing changes no file."""
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    heuristic = rescore(json.loads(HEURISTIC.read_text(encoding="utf-8")))
    HEURISTIC.write_text(_dump(heuristic), encoding="utf-8", newline="")
    before, after = heuristic["published_summary"], heuristic["summary"]
    print(
        f"{HEURISTIC.name}: {before['correct']}/{before['total']} correct as published, "
        f"{after['correct']}/{after['total']} once rescored"
    )

    ragas = resummarize_ragas(json.loads(RAGAS.read_text(encoding="utf-8")))
    RAGAS.write_text(_dump(ragas), encoding="utf-8", newline="")
    measured = [m for m in RAGAS_COLUMN if ragas["summary"][f"avg_{m}"] is not None]
    print(f"{RAGAS.name}: {len(measured)}/{len(RAGAS_COLUMN)} metrics measured on this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
