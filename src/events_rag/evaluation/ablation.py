"""Run every retrieval configuration over the evaluation set and collect the numbers.

A configuration that fails is reported with its error rather than dropped: the RAGAS
metrics that silently returned null are exactly the failure mode to avoid.
"""

import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from events_rag.evaluation.indexes import ChunkingVariant, chunks_for, index_for
from events_rag.evaluation.retrieval import aggregate, target_rank
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
        "per_question": None,
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
    # One rank per question, kept so that two configurations can be compared as the paired
    # runs they are. Averages alone cannot answer « which questions does it win », and the
    # published table was read for two weeks with the interval of a single proportion.
    result["per_question"] = [
        {"uid": row["target"], "rank": target_rank(row["retrieved"], row["target"])} for row in rows
    ]
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


def passages_by_uid(chunks: Sequence[Document]) -> dict[str, str]:
    """Reassemble every chunk of an event into the text the reranker is asked to score.

    The first version built ``{uid: doc.page_content for doc in chunks}``. With 2 046 chunks
    for 1 000 events, that dictionary keeps only the LAST chunk of every split event: for
    about half the corpus the cross-encoder scored a trailing block — usually dates and
    prices — instead of the event. The configuration did not fail; it returned a low,
    plausible number, and that number was published as a property of reranking.

    Chunks arrive in reading order, so joining them in iteration order restores the text
    that was indexed.
    """
    passages: dict[str, list[str]] = {}
    for doc in chunks:
        uid = str(doc.metadata.get("uid", ""))
        if uid:
            passages.setdefault(uid, []).append(doc.page_content)
    return {uid: "\n".join(parts) for uid, parts in passages.items()}


def _flashrank_scorer() -> Callable[[str, list[str]], list[float]]:
    """Score candidate uids against the query with a small cross-encoder.

    Imported lazily: the reranker lives in an optional dependency group, and the other
    configurations must run without it installed.
    """
    from flashrank import Ranker, RerankRequest

    # The model cache goes to the OS temp directory, not under data/. It is a downloaded
    # artefact, not project data, and on a synced folder the download raced its own
    # extraction: FlashRank opened the archive before the write was visible and failed
    # with "File is not a zip file" on a file that was in fact a valid zip.
    ranker = Ranker(cache_dir=str(Path(tempfile.gettempdir()) / "flashrank"))
    texts = passages_by_uid(chunks_for(BASELINE))

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
            "Narrow to the town named in the question. Read the gain with care: the "
            "questions are hand-written and name their town, because someone looking "
            "for an outing says where — but that choice favours this configuration."
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
