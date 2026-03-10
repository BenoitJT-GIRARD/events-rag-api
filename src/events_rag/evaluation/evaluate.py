import json
from pathlib import Path

from events_rag.config import get_settings
from events_rag.rag.service import answer_question


def load_reference_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    return payload


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0

    answer_lower = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    return hits / len(expected_keywords)


def city_match(sources: list[dict], expected_city: str | None) -> bool:
    if not expected_city:
        return True

    expected_city_lower = expected_city.lower()
    return any(
        (source.get("city") or "").lower() == expected_city_lower
        for source in sources
    )


def evaluate_case(case: dict) -> dict:
    result = answer_question(question=case["question"], top_k=5)
    answer = result["answer"]
    sources = result["sources"]

    coverage = keyword_coverage(answer, case.get("expected_keywords", []))
    has_city_match = city_match(sources, case.get("expected_city"))

    if coverage >= 0.66 and has_city_match:
        label = "correct"
    elif coverage > 0 or has_city_match:
        label = "partially_correct"
    else:
        label = "incorrect"

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "reference_answer": case["reference_answer"],
        "expected_keywords": case.get("expected_keywords", []),
        "keyword_coverage": round(coverage, 3),
        "city_match": has_city_match,
        "label": label,
        "sources": sources,
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for item in results if item["label"] == "correct")
    partially_correct = sum(1 for item in results if item["label"] == "partially_correct")
    incorrect = sum(1 for item in results if item["label"] == "incorrect")

    avg_keyword_coverage = (
        sum(item["keyword_coverage"] for item in results) / total if total else 0.0
    )

    return {
        "total": total,
        "correct": correct,
        "partially_correct": partially_correct,
        "incorrect": incorrect,
        "avg_keyword_coverage": round(avg_keyword_coverage, 3),
    }


def run_evaluation() -> dict:
    settings = get_settings()
    settings.eval_data_dir.mkdir(parents=True, exist_ok=True)

    input_path = settings.eval_data_dir / "reference_qa.json"
    output_path = settings.eval_data_dir / "evaluation_results.json"

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