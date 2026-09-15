"""Deterministic retrieval scoring: no judge, no second model, no API call to grade.

Each question records the `uid` of the event it was written from, so recall and MRR are
computed against a hard label. The refusal patterns are the other half: a question about an
event that does not exist is answered correctly by refusing, and a match on those patterns
is what counts as a refusal.
"""

import json
import re
from pathlib import Path

from events_rag.config import get_settings
from events_rag.rag.service import answer_question


def load_reference_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    return payload


def keyword_coverage(
    answer: str,
    expected_keywords: list[str],
    case_type: str = "positive",
) -> float | None:
    """Fraction of the expected keywords found in the answer.

    Returns None when a case defines no keywords, so those cases stay out of the
    aggregate instead of contributing a free 1.0. A positive case with no keywords
    is a dataset defect and still scores 0.0.
    """
    if not expected_keywords:
        return 0.0 if case_type == "positive" else None

    answer_lower = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    return hits / len(expected_keywords)


def city_match(sources: list[dict], expected_city: str | None) -> bool:
    if not expected_city:
        return True

    expected_city_lower = expected_city.lower()
    return any((source.get("city") or "").lower() == expected_city_lower for source in sources)


# Out-of-corpus refusals are matched on the negation, not on a fixed noun: the model
# reuses the noun from the question ("aucun festival", "aucune exposition"), so a list
# keyed on "événement" misses every real refusal.
REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\baucune?\b",
        r"\bne figure(?:nt)? pas\b",
        r"\bn['’]est pas (?:mentionn|disponible|pr[ée]cis)",
        r"\bil n['’]y a pas\b",
        r"\bje ne dispose pas\b",
        r"\bje n['’]ai pas\b",
        r"\bpas d['’](?:information|[ée]v[ée]nement|r[ée]sultat)",
        r"\bhors corpus\b",
        r"\bpas dans le corpus\b",
    )
]


def is_refusal(answer: str) -> bool:
    return any(pattern.search(answer) for pattern in REFUSAL_PATTERNS)


def evaluate_case(case: dict) -> dict:
    result = answer_question(question=case["question"], top_k=5)
    answer = result["answer"]
    sources = result["sources"]
    case_type = case.get("case_type", "positive")

    coverage = keyword_coverage(answer, case.get("expected_keywords", []), case_type)
    has_city_match = city_match(sources, case.get("expected_city"))

    if case_type == "negative":
        label = "correct" if is_refusal(answer) else "incorrect"
    elif coverage is not None and coverage >= 0.66 and has_city_match:
        label = "correct"
    elif (coverage or 0.0) > 0 or has_city_match:
        label = "partially_correct"
    else:
        label = "incorrect"

    ground_truth = case.get("ground_truth") or case.get("reference_answer", "")

    return {
        "id": case["id"],
        "case_type": case_type,
        "question": case["question"],
        "answer": answer,
        "ground_truth": ground_truth,
        "expected_keywords": case.get("expected_keywords", []),
        "keyword_coverage": round(coverage, 3) if coverage is not None else None,
        "city_match": has_city_match,
        "label": label,
        "sources": sources,
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for item in results if item["label"] == "correct")
    partially_correct = sum(1 for item in results if item["label"] == "partially_correct")
    incorrect = sum(1 for item in results if item["label"] == "incorrect")

    # Keyword coverage is only meaningful on positive cases. On a negative case the
    # expected keywords are the terms of the question itself, so any refusal that
    # restates the question scores 1.0 — an echo, not a correct answer. Averaging
    # across case types therefore mixes a real signal with a meaningless one.
    covered = [
        item["keyword_coverage"]
        for item in results
        if item.get("case_type", "positive") == "positive" and item["keyword_coverage"] is not None
    ]
    avg_keyword_coverage = round(sum(covered) / len(covered), 3) if covered else None

    by_case_type: dict[str, dict[str, int]] = {}
    for item in results:
        ct = item.get("case_type", "positive")
        bucket = by_case_type.setdefault(ct, {"total": 0})
        bucket["total"] += 1
        bucket[item["label"]] = bucket.get(item["label"], 0) + 1

    return {
        "total": total,
        "correct": correct,
        "partially_correct": partially_correct,
        "incorrect": incorrect,
        "avg_keyword_coverage_positive": avg_keyword_coverage,
        "keyword_coverage_n": len(covered),
        "by_case_type": by_case_type,
    }


def run_evaluation() -> dict:
    settings = get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    input_path = settings.questions_dir / "reference_qa.json"
    output_path = settings.reports_dir / "evaluation_results.json"

    dataset = load_reference_dataset(input_path)
    results = [evaluate_case(case) for case in dataset]
    summary = summarize(results)

    payload = {
        "summary": summary,
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return payload
