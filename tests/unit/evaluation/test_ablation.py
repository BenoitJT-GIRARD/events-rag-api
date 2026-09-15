"""The harness scores every configuration the same way, and hides none of them.

Three things it guards: only positive cases enter the retrieval scores, a configuration
that raises is reported with its error instead of vanishing from the table, and the
registry holds the configurations the README claims were tried, under unique names.
"""

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


def test_the_scorer_receives_the_whole_event_and_not_its_last_block():
    """An event split into several chunks reaches the cross-encoder with its title.

    The first version indexed `{uid: doc.page_content}`: over 2 046 chunks for 1 000 events,
    only the LAST chunk of every split event survived, and the cross-encoder scored a block of
    dates and prices. The published 0.55 against 0.90 measured that preparation, not
    reranking.
    """
    from langchain_core.documents import Document

    from events_rag.evaluation.ablation import passages_by_uid

    chunks = [
        Document(page_content="Concert de l'Orchestre national", metadata={"uid": "evt-1"}),
        Document(page_content="Le 4 mai 2026, 20 h. Tarif plein 18 €.", metadata={"uid": "evt-1"}),
        Document(page_content="Exposition Léger", metadata={"uid": "evt-2"}),
    ]

    passages = passages_by_uid(chunks)

    assert "Concert de l'Orchestre national" in passages["evt-1"]
    assert "Tarif plein 18 €" in passages["evt-1"]
    assert passages["evt-2"] == "Exposition Léger"


def test_a_chunk_without_a_uid_enters_no_passage():
    from langchain_core.documents import Document

    from events_rag.evaluation.ablation import passages_by_uid

    passages = passages_by_uid([Document(page_content="orphelin", metadata={})])

    assert passages == {}
