import pytest

from events_rag.evaluation.evaluate import (
    city_match,
    is_refusal,
    keyword_coverage,
    summarize,
)


def test_keyword_coverage_returns_fraction() -> None:
    answer = "Cet événement est gratuit à Montpellier et propose un concert."
    expected_keywords = ["gratuit", "Montpellier", "concert"]

    score = keyword_coverage(answer, expected_keywords)

    assert score == 1.0


def test_city_match_returns_true_when_source_city_matches() -> None:
    sources = [
        {
            "uid": "1",
            "title": "Concert",
            "city": "Montpellier",
            "date": "2026-03-10",
            "score": None,
        }
    ]

    assert city_match(sources, "Montpellier") is True


def test_summarize_counts_labels() -> None:
    results = [
        {"label": "correct", "keyword_coverage": 1.0},
        {"label": "partially_correct", "keyword_coverage": 0.5},
        {"label": "incorrect", "keyword_coverage": 0.0},
    ]

    summary = summarize(results)

    assert summary["total"] == 3
    assert summary["correct"] == 1
    assert summary["partially_correct"] == 1
    assert summary["incorrect"] == 1
    assert summary["avg_keyword_coverage_positive"] == 0.5
    assert summary["keyword_coverage_n"] == 3


def test_keyword_coverage_is_none_without_keywords_on_non_positive_case() -> None:
    assert keyword_coverage("peu importe", [], case_type="negative") is None
    assert keyword_coverage("peu importe", [], case_type="positive") == 0.0


def test_summarize_averages_positive_cases_only() -> None:
    # A refusal that restates the question scores 1.0 on a negative case; letting it
    # into the mean would report an echo as a success.
    results = [
        {"label": "correct", "keyword_coverage": 0.5, "case_type": "positive"},
        {"label": "correct", "keyword_coverage": 1.0, "case_type": "negative"},
    ]

    summary = summarize(results)

    assert summary["avg_keyword_coverage_positive"] == 0.5
    assert summary["keyword_coverage_n"] == 1
    assert summary["by_case_type"]["negative"] == {"total": 1, "correct": 1}


@pytest.mark.parametrize(
    "answer",
    [
        "D'après le contexte fourni, aucune information concernant des opéras n'est disponible.",
        "Aucun festival de cirque à Lyon en 2024 n'est mentionné.",
        "Aucune exposition d'art contemporain au Musée des Beaux-Arts de Lyon.",
        "Il n'y a pas d'événement correspondant dans le corpus.",
        "Je ne dispose pas de cette information.",
    ],
)
def test_is_refusal_matches_the_noun_used_by_the_model(answer: str) -> None:
    # The model reuses the noun from the question, so matching on "événement" alone
    # missed every real refusal and scored the five negative cases as failures.
    assert is_refusal(answer) is True


def test_is_refusal_rejects_a_confident_answer() -> None:
    assert is_refusal("Le concert a lieu le 12 mars à l'Opéra Comédie de Montpellier.") is False
