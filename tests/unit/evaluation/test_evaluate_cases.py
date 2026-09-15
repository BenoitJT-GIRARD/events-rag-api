"""How a single answer is labelled, and what the summary refuses to average.

The grader never calls a second model: a question carries the uid of the event it was written
from, and a refusal is recognised by its negation. Both halves are decided here, so both are
tested here, with the model replaced by a function that returns what the test wants to grade.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from events_rag.evaluation import evaluate


def answering(text: str, sources: list[dict] | None = None):
    def answer_question(question: str, top_k: int = 5) -> dict:
        return {"answer": text, "sources": sources or []}

    return answer_question


POSITIVE = {
    "id": "q1",
    "question": "Quel concert a lieu à Toulouse en mai ?",
    "case_type": "positive",
    "expected_keywords": ["orchestre", "Toulouse", "mai"],
    "expected_city": "Toulouse",
    "ground_truth": "Un concert de l'orchestre national.",
}


def test_an_answer_carrying_the_keywords_and_the_town_is_correct(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate,
        "answer_question",
        answering("L'orchestre joue à Toulouse en mai.", [{"city": "Toulouse"}]),
    )

    result = evaluate.evaluate_case(POSITIVE)

    assert result["label"] == "correct"
    assert result["keyword_coverage"] == 1.0
    assert result["city_match"] is True


def test_the_right_town_alone_is_only_partly_correct(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate, "answer_question", answering("Un événement s'y tient.", [{"city": "Toulouse"}])
    )

    assert evaluate.evaluate_case(POSITIVE)["label"] == "partially_correct"


def test_a_refusal_answers_a_question_about_an_event_that_does_not_exist(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate, "answer_question", answering("Aucun festival de ce nom dans le corpus.")
    )
    negative = {**POSITIVE, "case_type": "negative", "expected_keywords": ["festival"]}

    assert evaluate.evaluate_case(negative)["label"] == "correct"


def test_an_invented_answer_to_an_impossible_question_is_incorrect(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluate, "answer_question", answering("Le festival a lieu le 3 juin à Albi.")
    )
    negative = {**POSITIVE, "case_type": "negative", "expected_keywords": ["festival"]}

    assert evaluate.evaluate_case(negative)["label"] == "incorrect"


@pytest.mark.parametrize(
    "answer",
    [
        "Aucun événement ne correspond.",
        "Cette information ne figure pas dans le corpus.",
        "Je ne dispose pas de cette information.",
        "Il n'y a pas d'exposition de ce type.",
        "La question est hors corpus.",
    ],
)
def test_a_refusal_is_recognised_by_its_negation_not_by_a_fixed_noun(answer: str) -> None:
    """The model reuses the noun of the question, so a list keyed on one noun misses most."""
    assert evaluate.is_refusal(answer)


def test_an_answer_that_names_an_event_is_not_a_refusal() -> None:
    assert not evaluate.is_refusal("Le concert de l'orchestre a lieu le 4 mai.")


def test_a_positive_case_with_no_keywords_scores_zero_instead_of_vanishing() -> None:
    assert evaluate.keyword_coverage("peu importe", [], "positive") == 0.0


def test_a_negative_case_with_no_keywords_stays_out_of_the_average() -> None:
    assert evaluate.keyword_coverage("peu importe", [], "negative") is None


def test_the_summary_averages_coverage_over_positive_cases_only() -> None:
    """On a negative case the keywords are the question's own words: an echo scores 1.0."""
    results = [
        {"label": "correct", "case_type": "positive", "keyword_coverage": 0.5},
        {"label": "correct", "case_type": "negative", "keyword_coverage": 1.0},
    ]

    summary = evaluate.summarize(results)

    assert summary["avg_keyword_coverage_positive"] == 0.5
    assert summary["keyword_coverage_n"] == 1
    assert summary["by_case_type"]["negative"]["total"] == 1


def test_the_run_writes_its_results_where_the_settings_say(tmp_path: Path, monkeypatch) -> None:
    questions = tmp_path / "questions"
    reports = tmp_path / "reports"
    questions.mkdir()
    (questions / "reference_qa.json").write_text(json.dumps([POSITIVE]), encoding="utf-8")

    settings = evaluate.get_settings()
    monkeypatch.setattr(settings, "questions_dir", questions)
    monkeypatch.setattr(settings, "reports_dir", reports)
    monkeypatch.setattr(
        evaluate,
        "answer_question",
        answering("L'orchestre joue à Toulouse en mai.", [{"city": "Toulouse"}]),
    )

    payload = evaluate.run_evaluation()

    written = json.loads((reports / "evaluation_results.json").read_text(encoding="utf-8"))
    assert written["summary"]["total"] == 1
    assert payload["results"][0]["label"] == "correct"


def test_a_dataset_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "reference_qa.json"
    path.write_text(json.dumps({"cases": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON list"):
        evaluate.load_reference_dataset(path)
