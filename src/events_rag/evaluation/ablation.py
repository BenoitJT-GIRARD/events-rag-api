"""Run every retrieval configuration over the evaluation set and collect the numbers.

A configuration that fails is reported with its error rather than dropped: the RAGAS
metrics that silently returned null are exactly the failure mode to avoid.
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from events_rag.config import get_settings
from events_rag.evaluation.indexes import ChunkingVariant, chunks_for, index_for
from events_rag.evaluation.retrieval import aggregate
from events_rag.evaluation.strategies import (
    BM25Search,
    CityFilter,
    DenseSearch,
    HybridSearch,
    Rerank,
    SearchStrategy,
)
from events_rag.logger import get_logger

logger = get_logger(__name__)

BASELINE = ChunkingVariant("baseline", 800, 120)
ONE_CHUNK = ChunkingVariant("one-chunk-per-event", None, None)
WITH_HEADER = ChunkingVariant("metadata-header", 800, 120, header=True)


@dataclass(frozen=True)
class AblationConfig:
    name: str
    variant: ChunkingVariant
    build: Callable[[], SearchStrategy]
    note: str = field(default="")


def run_config(config: AblationConfig, cases: list[dict], k: int = 10) -> dict:
    targets = [case for case in cases if case.get("source_uid")]
    result: dict = {
        "name": config.name,
        "variant": config.variant.name,
        "note": config.note,
        "metrics": None,
        "median_latency_ms": None,
        "error": None,
    }

    try:
        strategy = config.build()
        rows: list[dict] = []
        latencies: list[float] = []
        for case in targets:
            started = time.perf_counter()
            retrieved = strategy.search(case["question"], k)
            latencies.append((time.perf_counter() - started) * 1000)
            rows.append({"target": case["source_uid"], "retrieved": retrieved})
    except Exception as exc:  # noqa: BLE001 - a broken config must not stop the run
        logger.warning("ablation.config_failed", name=config.name, error=str(exc))
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["metrics"] = aggregate(rows)
    result["median_latency_ms"] = round(statistics.median(latencies), 1) if latencies else None
    return result


def run_all(configs: list[AblationConfig], cases: list[dict]) -> list[dict]:
    return [run_config(config, cases) for config in configs]


def _city_by_uid() -> dict[str, str]:
    """Map each event uid to its town, for the city filter."""
    return {
        str(doc.metadata.get("uid", "")): str(doc.metadata.get("city") or "")
        for doc in chunks_for(BASELINE)
        if doc.metadata.get("uid")
    }


def _flashrank_scorer() -> Callable[[str, list[str]], list[float]]:
    """Score candidate uids against the query with a small cross-encoder.

    Imported lazily: the reranker lives in an optional dependency group, and the other
    configurations must run without it installed.
    """
    from flashrank import Ranker, RerankRequest

    settings = get_settings()
    ranker = Ranker(cache_dir=str(settings.data_dir / "flashrank"))
    texts = {str(doc.metadata.get("uid", "")): doc.page_content for doc in chunks_for(BASELINE)}

    def score(query: str, uids: list[str]) -> list[float]:
        passages = [{"id": uid, "text": texts.get(uid, "")} for uid in uids]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        by_uid = {item["id"]: float(item["score"]) for item in ranked}
        return [by_uid.get(uid, 0.0) for uid in uids]

    return score


CONFIGS: list[AblationConfig] = [
    AblationConfig(
        "bm25-only",
        BASELINE,
        lambda: BM25Search(chunks_for(BASELINE)),
        note="Trivial floor: no embeddings at all.",
    ),
    AblationConfig(
        "dense-baseline",
        BASELINE,
        lambda: DenseSearch(index_for(BASELINE)),
        note="The shipped configuration.",
    ),
    AblationConfig(
        "dense-one-chunk-per-event",
        ONE_CHUNK,
        lambda: DenseSearch(index_for(ONE_CHUNK)),
        note="Index each event whole instead of splitting it.",
    ),
    AblationConfig(
        "dense-metadata-header",
        WITH_HEADER,
        lambda: DenseSearch(index_for(WITH_HEADER)),
        note="Embed title, venue, city and date with the text so proper nouns match.",
    ),
    AblationConfig(
        "dense+city-filter",
        BASELINE,
        lambda: CityFilter(DenseSearch(index_for(BASELINE)), _city_by_uid()),
        note=(
            "Narrow to the town named in the question. Part of any gain may be an "
            "artefact: questions were generated from an event whose town the prompt "
            "showed the model."
        ),
    ),
    AblationConfig(
        "hybrid-rrf",
        BASELINE,
        lambda: HybridSearch(DenseSearch(index_for(BASELINE)), BM25Search(chunks_for(BASELINE))),
        note="Reciprocal rank fusion of dense and lexical search.",
    ),
    AblationConfig(
        "hybrid-rrf+rerank",
        BASELINE,
        lambda: Rerank(
            HybridSearch(DenseSearch(index_for(BASELINE)), BM25Search(chunks_for(BASELINE))),
            _flashrank_scorer(),
        ),
        note="Cross-encoder reranking on top of hybrid search. Optional dependency, 41 MB.",
    ),
]
