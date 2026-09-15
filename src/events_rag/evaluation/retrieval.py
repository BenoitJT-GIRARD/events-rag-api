"""Deterministic retrieval metrics, scored against the event a question came from.

Unlike the RAGAS metrics these need no LLM judge: the evaluation set records the uid of
the event each positive question was generated from, which is exactly the relevance label
a retrieval benchmark needs.
"""

import math


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


def target_rank(retrieved: list[str], target: str) -> int | None:
    """The 1-based rank of the target event, or None when it was not retrieved.

    One number per question and per configuration, which is what a PAIRED comparison needs:
    the seven configurations answer the same twenty questions, so what separates two of them
    is the set of questions where one succeeds and the other fails, and that set is invisible
    once each side is averaged into a proportion.
    """
    ranked = dedupe_uids(retrieved)
    if target not in ranked:
        return None
    return ranked.index(target) + 1


def mcnemar_exact(first: list[bool], second: list[bool]) -> dict:
    """The exact McNemar test on two runs over the same questions.

    Only the discordant pairs carry information: a question both configurations get right,
    or both get wrong, says nothing about which is better. Under the null hypothesis the
    discordant pairs split like a fair coin, so the two-sided p-value is the binomial tail.

    Returns the two discordant counts and the p-value. With twenty questions the test is
    almost always undecided, and saying so is the point of computing it.
    """
    if len(first) != len(second):
        raise ValueError("a paired test needs the same questions on both sides")
    only_first = sum(1 for a, b in zip(first, second, strict=True) if a and not b)
    only_second = sum(1 for a, b in zip(first, second, strict=True) if b and not a)
    discordant = only_first + only_second
    if discordant == 0:
        return {"only_first": 0, "only_second": 0, "discordant": 0, "p_value": 1.0}

    smaller = min(only_first, only_second)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1))
    p_value = min(1.0, 2 * tail / (2**discordant))
    return {
        "only_first": only_first,
        "only_second": only_second,
        "discordant": discordant,
        "p_value": round(p_value, 4),
    }


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
