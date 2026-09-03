import pytest

from events_rag.evaluation.ablation import CONFIGS, AblationConfig, run_config
from events_rag.evaluation.indexes import ChunkingVariant

VARIANT = ChunkingVariant("test", 800, 120)


class FakeStrategy:
    def __init__(self, uids: list[str]) -> None:
        self._uids = uids

    def search(self, query: str, k: int) -> list[str]:
        return self._uids[:k]


def _cases() -> list[dict]:
    return [
        {"id": "q1", "case_type": "positive", "question": "a?", "source_uid": "a"},
        {"id": "q2", "case_type": "negative", "question": "b?"},
    ]


def test_run_config_scores_positive_cases_only() -> None:
    config = AblationConfig("fake", VARIANT, lambda: FakeStrategy(["a", "z"]))

    result = run_config(config, _cases())

    assert result["error"] is None
    assert result["metrics"]["n"] == 1
    assert result["metrics"]["recall@1"] == 1.0


def test_run_config_reports_a_failure_instead_of_hiding_it() -> None:
    def explode() -> FakeStrategy:
        raise RuntimeError("index missing")

    result = run_config(AblationConfig("broken", VARIANT, explode), _cases())

    assert result["metrics"] is None
    assert "index missing" in result["error"]


def test_run_config_records_a_latency() -> None:
    config = AblationConfig("fake", VARIANT, lambda: FakeStrategy(["a"]))

    result = run_config(config, _cases())

    assert result["median_latency_ms"] >= 0


def test_registry_names_are_unique() -> None:
    names = [config.name for config in CONFIGS]

    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "expected",
    ["bm25-only", "dense-baseline", "dense-one-chunk-per-event", "hybrid-rrf"],
)
def test_registry_contains_the_planned_configurations(expected: str) -> None:
    assert expected in {config.name for config in CONFIGS}
