from events_rag.evaluation.evaluate import city_match, keyword_coverage, summarize


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
    assert summary["avg_keyword_coverage"] == 0.5