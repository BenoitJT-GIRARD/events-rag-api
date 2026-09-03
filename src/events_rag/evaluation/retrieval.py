"""Deterministic retrieval metrics, scored against the event a question came from.

Unlike the RAGAS metrics these need no LLM judge: the evaluation set records the uid of
the event each positive question was generated from, which is exactly the relevance label
a retrieval benchmark needs.
"""


def dedupe_uids(uids: list[str]) -> list[str]:
    """Keep the first occurrence of each uid, preserving rank order.

    One event is split into several chunks, so a single event can occupy several
    positions of a result list. Ranking without deduplication inflates recall.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for uid in uids:
        if uid not in seen:
            seen.add(uid)
            ordered.append(uid)
    return ordered


def recall_at_k(retrieved: list[str], target: str, k: int) -> float:
    return 1.0 if target in dedupe_uids(retrieved)[:k] else 0.0


def reciprocal_rank(retrieved: list[str], target: str, k: int = 10) -> float:
    ranked = dedupe_uids(retrieved)[:k]
    if target not in ranked:
        return 0.0
    return 1.0 / (ranked.index(target) + 1)


def aggregate(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Aggregate per-question results. `rows` carries only cases that have a target."""
    if not rows:
        return {"n": 0, "mrr@10": None} | {f"recall@{k}": None for k in ks}

    summary: dict = {"n": len(rows)}
    for k in ks:
        scores = [recall_at_k(row["retrieved"], row["target"], k) for row in rows]
        summary[f"recall@{k}"] = round(sum(scores) / len(scores), 3)

    ranks = [reciprocal_rank(row["retrieved"], row["target"]) for row in rows]
    summary["mrr@10"] = round(sum(ranks) / len(ranks), 3)
    return summary
